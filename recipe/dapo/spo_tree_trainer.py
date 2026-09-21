# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""SPO-Tree policy iteration with DAPO prompt-level dynamic sampling."""

import json
import os
from collections import defaultdict
from dataclasses import asdict
from time import perf_counter

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer

from .spo_tree_core import SPOTreeConfig
from .spo_tree_rollout import SPOTreeRollout, make_training_batch


def validate_spo_config(config) -> SPOTreeConfig:
    spo = SPOTreeConfig(**dict(config.algorithm.spo_tree))
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    if str(actor.strategy) not in {"fsdp", "fsdp2"} or config.trainer.get("use_legacy_worker_impl") == "disable":
        raise ValueError("SPO-Tree requires the legacy FSDP/FSDP2 worker API")
    if str(rollout.name) != "vllm" or rollout.multi_turn.enable:
        raise ValueError("SPO-Tree requires single-turn vLLM rollouts")
    if rollout.get("skip_rollout", False) or rollout.get("enable_rollout_routing_replay", False):
        raise ValueError("SPO-Tree does not support rollout skipping or MoE routing replay")
    if int(actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError("SPO-Tree currently requires sequence_parallel_size=1")
    if actor.get("use_prefix_grouper", False) or actor.get("use_fused_kernels", False):
        raise ValueError("SPO-Tree requires prefix grouping and fused kernels disabled")
    if config.algorithm.get("sparse_counterfactual_credit", {}).get("enabled", False):
        raise ValueError("SPO-Tree must not be combined with sparse counterfactual credit")
    if actor.get("counterfactual_credit_head", {}).get("enabled", False):
        raise ValueError("SPO-Tree does not use a probe")
    if config.algorithm.get("step_split", {}).get("enabled", False):
        raise ValueError("SPO-Tree uses fixed token segments, not semantic step splitting")
    if config.algorithm.use_kl_in_reward or actor.use_kl_loss:
        raise ValueError("This DAPO-aligned SPO baseline disables reference KL in both reward and loss")
    if config.reward.reward_model.enable:
        raise ValueError("SPO-Tree currently supports rule-based verifiers only")
    if config.algorithm.adv_estimator != "spo_tree":
        raise ValueError("SPO-Tree requires algorithm.adv_estimator=spo_tree")
    if int(rollout.n) != spo.max_leaves:
        raise ValueError("rollout.n must equal the product of SPO branching factors (maximum leaf budget)")
    if actor.entropy_coeff != 0 or actor.loss_agg_mode != "token-mean":
        raise ValueError("SPO uses masked token-mean policy loss without an entropy bonus")
    if actor.policy_loss.loss_mode != "vanilla":
        raise ValueError("This SPO baseline uses DAPO's vanilla token-level policy loss")
    if int(actor.ppo_epochs) != 1:
        raise ValueError("This on-policy SPO baseline uses one PPO epoch per sampled batch")
    if config.trainer.balance_batch:
        raise ValueError("SPO keeps prompt groups together; set trainer.balance_batch=False")
    if str(rollout.checkpoint_engine.backend) != "naive":
        raise ValueError("SPO currently requires the naive hybrid checkpoint engine")
    filters = config.algorithm.filter_groups
    if not filters.enable or filters.metric not in {"acc", "seq_reward"}:
        raise ValueError("SPO requires dynamic filtering with metric=seq_reward or acc")
    if filters.get("fill_shortfall", False) or filters.get("adaptive_topup", {}).get("enabled", False):
        raise ValueError("SPO does not use fill-shortfall or adaptive top-up sampling")
    correction = config.algorithm.get("rollout_correction", {}) or {}
    if correction.get("bypass_mode", False) or correction.get("rollout_is") is not None:
        raise ValueError("SPO uses frozen actor log probabilities, without extra rollout IS weights")
    world_size = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    prompt_mini = int(actor.ppo_mini_batch_size)
    prompt_batch = int(config.data.train_batch_size)
    if prompt_batch % prompt_mini or prompt_mini % world_size:
        raise ValueError("SPO requires train_batch_size divisible by prompt mini-batch, itself divisible by DP size")
    if int(config.data.gen_batch_size) < 1:
        raise ValueError("SPO gen_batch_size must be positive")
    if config.actor_rollout_ref.model.get("lora", {}).get("rank", 0) > 0:
        raise ValueError("This SPO baseline currently supports full-parameter training, not LoRA")
    return spo


def select_training_trees(trees, metric: str):
    """DAPO filtering uses actual terminal rewards, never padded/duplicated leaves."""
    selected = []
    passed = 0
    for tree in trees:
        values = [node.correctness if metric == "acc" else node.value for node in tree.leaves]
        if len(values) > 1 and any(value != values[0] for value in values[1:]):
            passed += 1
            selected.append(tree)
    return selected, passed


def tree_metrics(trees, max_response_length: int) -> dict[str, float]:
    leaves = [node for tree in trees for node in tree.leaves]
    lengths = np.asarray([len(node.response_ids) for node in leaves], dtype=np.float64)
    rewards = np.asarray([node.value for node in leaves], dtype=np.float64)
    result = {
        "spo/leaves_per_prompt": len(leaves) / len(trees),
        "spo/segments_per_prompt": sum(len(tree.nodes) - 1 for tree in trees) / len(trees),
        "spo/trainable_segments_per_prompt": sum(len(tree.training_nodes) for tree in trees) / len(trees),
        "response_length/mean": float(lengths.mean()),
        "response_length/max": float(lengths.max()),
        "response_length/min": float(lengths.min()),
        "response_length/clip_ratio": float((lengths == max_response_length).mean()),
        "critic/score/mean": float(rewards.mean()),
        "critic/score/max": float(rewards.max()),
        "critic/score/min": float(rewards.min()),
        "train/acc": float(np.mean([node.correctness for node in leaves])),
    }
    return result


class RaySPOTreeTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spo_config = validate_spo_config(self.config)
        if self.use_critic:
            raise ValueError("SPO-Tree estimates values from the tree and must not create a critic")
        self.spo_epoch = 0

    def init_workers(self):
        # The normal FSDP worker multiplies prompt_mini_batch_size by rollout_n.
        # Use node slots here, but keep rollout.n as the independent leaf budget.
        with open_dict(self.config.actor_rollout_ref.actor):
            self.config.actor_rollout_ref.actor.rollout_n = self.spo_config.max_nodes
        super().init_workers()

    def _save_trainer_extra_state(self, local_global_step_folder: str) -> None:
        state = {"epoch": self.spo_epoch, "spo_tree": asdict(self.spo_config)}
        with open(os.path.join(local_global_step_folder, "spo_tree_state.json"), "w") as stream:
            json.dump(state, stream)

    def _load_trainer_extra_state(self, global_step_folder: str) -> None:
        path = os.path.join(global_step_folder, "spo_tree_state.json")
        if not os.path.isfile(path):
            raise ValueError("Missing SPO trainer state; do not resume a DAPO/S2D training checkpoint as SPO")
        with open(path) as stream:
            state = json.load(stream)
        restored = SPOTreeConfig(**state["spo_tree"])
        if restored != self.spo_config:
            raise ValueError("SPO configuration differs from the checkpoint; use a separate experiment directory")
        self.spo_epoch = int(state["epoch"])

    def _train_trees(self, trees, timing):
        batch = make_training_batch(trees, self.spo_config, self.tree_rollout.pad_token_id)
        batch.meta_info.update(
            temperature=float(self.config.actor_rollout_ref.rollout.temperature),
            pad_token_id=self.tree_rollout.pad_token_id,
            global_steps=self.global_steps,
            calculate_entropy=False,
            global_token_num=batch.batch["attention_mask"].sum(dim=-1).tolist(),
        )
        with marked_timer("old_log_prob", timing, "blue"):
            batch = batch.union(self.actor_rollout_wg.compute_log_prob(batch))
        with marked_timer("update_actor", timing, "red"):
            output = self._update_actor(batch)
        return reduce_metrics(output.meta_info["metrics"])

    def _finish(self, logger, progress, *, last_saved_step, last_validated_step):
        """Finalize even when dynamic sampling exhausts the data before max steps."""
        timing = defaultdict(float)
        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
        folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if self.global_steps > 0 and (last_saved_step != self.global_steps or not os.path.isdir(folder)):
            with marked_timer("save_checkpoint", timing):
                # Checkpoints must run in training mode, never with vLLM's KV
                # cache occupying the shared GPU memory.
                self.checkpoint_manager.sleep_replicas()
                self._save_checkpoint()
                self.checkpoint_manager.update_weights(self.global_steps)
        elif self.global_steps > 0:
            # Additional filtered batches may have advanced the dataloader
            # after a periodic save, without advancing the optimizer. Refresh
            # only driver state; the saved actor/optimizer are already current.
            torch.save(self.train_dataloader.state_dict(), os.path.join(folder, "data.pt"))
            self._save_trainer_extra_state(folder)
        metrics = {}
        if last_validated_step != self.global_steps:
            with marked_timer("testing", timing):
                metrics.update(self._validate())
        metrics.update({f"timing_s/{key}": value for key, value in timing.items()})
        if metrics:
            logger.log(data=metrics, step=self.global_steps)
        progress.close()

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        self.tree_rollout = SPOTreeRollout(
            self.tokenizer, self.config, self.async_rollout_manager, self.reward_loop_manager
        )
        last_saved_step = self.global_steps if self.global_steps else -1
        last_validated_step = -1
        if self.config.trainer.get("val_before_train", True) or self.config.trainer.get("val_only", False):
            logger.log(data=self._validate(), step=self.global_steps)
            last_validated_step = self.global_steps
            if self.config.trainer.get("val_only", False):
                return
        progress = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="SPO-Tree training")
        pending = []  # Only the current optimizer batch, NEVER a cross-update replay buffer.
        timing = defaultdict(float)
        num_gen_batches = num_prompt_pass_filter = num_generated_tokens = num_generated_prompts = 0
        step_started = perf_counter()
        for epoch in range(self.spo_epoch, int(self.config.trainer.total_epochs)):
            self.spo_epoch = epoch
            if self.global_steps >= self.total_training_steps:
                break
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                num_gen_batches += 1
                source = DataProto.from_single_dict(batch_dict)
                with marked_timer("gen", timing, "red"):
                    trees = self.tree_rollout.prepare_prompts(source)
                    trees = self.tree_rollout.generate(
                        trees, global_step=self.global_steps + 1, generation_batch=num_gen_batches
                    )
                with marked_timer("reward", timing, "yellow"):
                    self.tree_rollout.score(trees)
                num_generated_tokens += sum(len(node.segment_ids) for tree in trees for node in tree.nodes[1:])
                num_generated_prompts += len(trees)
                selected, passed = select_training_trees(trees, self.config.algorithm.filter_groups.metric)
                num_prompt_pass_filter += passed
                pending.extend(selected)
                prompt_batch_size = int(self.config.data.train_batch_size)
                if len(pending) < prompt_batch_size:
                    limit = int(self.config.algorithm.filter_groups.max_num_gen_batches)
                    if limit > 0 and num_gen_batches >= limit:
                        raise ValueError(
                            f"SPO dynamic sampling kept {len(pending)}/{prompt_batch_size} prompts after "
                            f"{num_gen_batches} batches. Increase gen_batch_size or max_num_gen_batches."
                        )
                    print(f"SPO dynamic sampling: {len(pending)}/{prompt_batch_size} prompts, generating again")
                    continue
                training_trees = pending[:prompt_batch_size]
                pending = []  # Discard surplus prompts as in DAPO; no reuse after policy update.
                self.checkpoint_manager.sleep_replicas()
                self.global_steps += 1
                metrics = tree_metrics(training_trees, int(self.config.data.max_response_length))
                metrics.update(self._train_trees(training_trees, timing))
                final_step = self.global_steps >= self.total_training_steps
                save_freq = int(self.config.trainer.save_freq)
                if final_step or (save_freq > 0 and self.global_steps % save_freq == 0):
                    with marked_timer("save_checkpoint", timing, "green"):
                        self._save_checkpoint()
                    last_saved_step = self.global_steps
                with marked_timer("update_weights", timing, "red"):
                    self.checkpoint_manager.update_weights(self.global_steps)
                # Match DAPO: step time includes checkpoint saving, but not validation.
                step_duration = perf_counter() - step_started
                test_freq = int(self.config.trainer.test_freq)
                if final_step or (test_freq > 0 and self.global_steps % test_freq == 0):
                    with marked_timer("testing", timing, "green"):
                        metrics.update(self._validate())
                    last_validated_step = self.global_steps
                metrics.update({f"timing_s/{key}": value for key, value in timing.items()})
                metrics.update(
                    {
                        "timing_s/step": step_duration,
                        "train/num_gen_batches": num_gen_batches,
                        "train/num_prompt_pass_filter": num_prompt_pass_filter,
                        "spo/generated_tokens": num_generated_tokens,
                        "spo/generated_prompts": num_generated_prompts,
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                logger.log(data=metrics, step=self.global_steps)
                progress.update(1)
                timing = defaultdict(float)
                num_gen_batches = num_prompt_pass_filter = num_generated_tokens = num_generated_prompts = 0
                step_started = perf_counter()
                if final_step:
                    break
            if self.global_steps >= self.total_training_steps:
                # A step limit can stop mid-epoch. Preserve that epoch together
                # with its dataloader cursor when resuming with a larger limit.
                break
            self.spo_epoch = epoch + 1
        self._finish(logger, progress, last_saved_step=last_saved_step, last_validated_step=last_validated_step)
