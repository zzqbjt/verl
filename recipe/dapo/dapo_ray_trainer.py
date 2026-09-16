# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from omegaconf import open_dict
from recipe.dapo.sparse_counterfactual_credit import SparseCounterfactualCreditSupervisor
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.sparse_counterfactual_credit import (
    build_credit_residual,
    credit_advantage_coefficient,
    merge_anchor_credit,
)
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def _save_trainer_extra_state(self, local_global_step_folder: str) -> None:
        super()._save_trainer_extra_state(local_global_step_folder)
        with open(os.path.join(local_global_step_folder, "dapo_trainer_state.json"), "w") as stream:
            json.dump({"data_epoch": self._data_epoch, "gen_steps": self.gen_steps}, stream)

    def _load_trainer_extra_state(self, global_step_folder: str) -> None:
        super()._load_trainer_extra_state(global_step_folder)
        path = os.path.join(global_step_folder, "dapo_trainer_state.json")
        if os.path.exists(path):
            with open(path) as stream:
                state = json.load(stream)
            self._data_epoch = int(state["data_epoch"])
            self.gen_steps = int(state["gen_steps"])
            return

        # Legacy checkpoints did not save the data epoch. At an exact boundary,
        # the last batch can be saved before the iterator sees StopIteration.
        # Resume that iterator in the preceding epoch, rather than consuming
        # the next epoch with an empty iteration.
        state = self.train_dataloader.state_dict()
        epoch_size = len(self.train_dataloader)
        yielded = state.get("_num_yielded")
        if yielded is None:
            yielded = state.get("_snapshot", {}).get("_snapshot_step", 0) + state.get(
                "_steps_since_snapshot", 0
            )
        epoch = self.global_steps // epoch_size
        if self.global_steps > 0 and self.global_steps % epoch_size == 0 and not state.get(
            "_iterator_finished", False
        ) and yielded >= epoch_size:
            epoch -= 1
        self._data_epoch = epoch
        print(
            f"Legacy DAPO checkpoint: inferred data_epoch={epoch}, batches_yielded={yielded}. "
            "Legacy checkpoints do not record exact epoch counts under dynamic sampling."
        )

    def _iter_training_batches(self):
        # Keep the epoch synchronized with the *data iterator*, not optimizer
        # steps: dynamic filtering may consume several batches per update.
        for epoch in range(self._data_epoch, self.config.trainer.total_epochs):
            self._data_epoch = epoch
            for batch_dict in self.train_dataloader:
                yield epoch, batch_dict
            self._data_epoch = epoch + 1

    def init_workers(self):
        credit_config = self.config.algorithm.get("sparse_counterfactual_credit", None) or {}
        actor_rollout_config = self.config.actor_rollout_ref
        training_rollout_n = int(actor_rollout_config.rollout.n)
        if credit_config.get("enabled", False) and credit_config.get("train_mc_branches", False):
            training_rollout_n += int(credit_config.branch_groups_per_prompt) * int(credit_config.num_samples)
        # Keep rollout.n unchanged: it controls base generation and outcome groups.
        # actor.rollout_n counts all training responses per prompt, including branches.
        # Always recompute so reinitialization or disabling branches cannot keep a stale count.
        with open_dict(actor_rollout_config.actor):
            actor_rollout_config.actor.rollout_n = training_rollout_n
        if credit_config.get("enabled", False) and not credit_config.get("use_probe", True):
            # Apply before constructing actors, including when resuming a full-method checkpoint.
            self.config.actor_rollout_ref.actor.counterfactual_credit_head.enabled = False
        super().init_workers()

    def _predict_counterfactual_credit(
        self, batch: DataProto, credit_coef: float, metrics: dict, timing_raw: dict
    ) -> DataProto:
        """Predict before fitting the probe, or leave only MC anchors for the ablation."""

        credit_config = self.config.algorithm.sparse_counterfactual_credit
        if not credit_config.get("use_probe", True):
            # merge_anchor_credit will fill the observed anchors later. Do not
            # infer any neighboring credits from known Q/V or boundary labels.
            batch.batch["credit_predictions"] = torch.zeros_like(batch.batch["credit_anchor_targets"])
            return batch

        with marked_timer("counterfactual_credit_head", timing_raw, "purple"):
            batch.meta_info["global_steps"] = self.global_steps
            batch.meta_info["credit_value_loss_weights"] = (
                float(credit_config.mc_value_loss_weight),
                float(credit_config.start_value_loss_weight),
                float(credit_config.terminal_value_loss_weight),
            )
            # When warmup makes lambda zero, skip dense policy predictions
            # while still fitting enabled MC, start, and terminal value sources.
            batch.meta_info["predict_all_credit_steps"] = credit_coef != 0.0
            credit_output = self.actor_rollout_wg.update_counterfactual_credit(batch)
            batch = batch.union(credit_output)
            metrics["credit/head_value_bce"] = float(
                credit_output.batch["credit_head_value_bce"].float().mean().item()
            )
            metrics["credit/head_value_mae"] = float(
                credit_output.batch["credit_head_value_mae"].float().mean().item()
            )
            for source_name in ("mc", "start", "terminal"):
                source_value_bce = float(
                    credit_output.batch[f"credit_head_value_bce_{source_name}"].float().mean().item()
                )
                source_value_mae = float(
                    credit_output.batch[f"credit_head_value_mae_{source_name}"].float().mean().item()
                )
                if np.isfinite(source_value_bce):
                    metrics[f"credit/head_value_bce/{source_name}"] = source_value_bce
                if np.isfinite(source_value_mae):
                    metrics[f"credit/head_value_mae/{source_name}"] = source_value_mae
            head_direction_agreement = float(
                credit_output.batch["credit_head_direction_agreement"].float().mean().item()
            )
            if np.isfinite(head_direction_agreement):
                metrics["credit/head_direction_agreement"] = head_direction_agreement
            head_direction_majority_baseline = float(
                credit_output.batch["credit_head_direction_majority_baseline"].float().mean().item()
            )
            if np.isfinite(head_direction_majority_baseline):
                metrics["credit/head_direction_majority_baseline"] = head_direction_majority_baseline
            head_direction_excess = float(
                credit_output.batch["credit_head_direction_excess"].float().mean().item()
            )
            if np.isfinite(head_direction_excess):
                metrics["credit/head_direction_excess"] = head_direction_excess
        return batch

    def _prepare_auxiliary_rewards(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
    ) -> DataProto:
        """Recipe hook for online reward models that run after dynamic sampling."""

        return batch

    def _augment_advantages(self, batch: DataProto, metrics: dict) -> DataProto:
        """Recipe hook for combining auxiliary returns with DAPO advantages."""

        return batch

    @staticmethod
    def _compute_branch_response_length_metrics(
        batch: DataProto,
        *,
        mc_branch_max_response_length: int,
    ) -> dict[str, float]:
        """Report full base lengths and newly generated MC continuation lengths."""

        if "is_mc_branch" not in batch.batch:
            return {}
        response_width = batch.batch["responses"].shape[-1]
        is_mc_branch = batch.batch["is_mc_branch"].bool()
        if is_mc_branch.shape != (len(batch),):
            raise ValueError("is_mc_branch must contain one boolean per response.")
        if not is_mc_branch.any():
            return {}
        if not 1 <= mc_branch_max_response_length <= response_width:
            raise ValueError("MC branch response limit must be in [1, padded response width].")

        full_response_lengths = batch.batch["attention_mask"][:, -response_width:].sum(dim=-1).float()
        if "credit_prefix_lengths" not in batch.batch:
            raise ValueError("MC branch response-length metrics require credit_prefix_lengths.")
        prefix_lengths = batch.batch["credit_prefix_lengths"].to(full_response_lengths.device).float()
        if prefix_lengths.shape != (len(batch),):
            raise ValueError("credit_prefix_lengths must contain one length per response.")

        base_lengths = full_response_lengths[~is_mc_branch]
        mc_prefix_lengths = prefix_lengths[is_mc_branch]
        mc_lengths = full_response_lengths[is_mc_branch] - mc_prefix_lengths
        mc_clip_lengths = mc_branch_max_response_length - mc_prefix_lengths
        metrics = {}
        for prefix, lengths, clip_lengths in (
            ("response_length/base", base_lengths, response_width),
            ("response_length/mc_branch", mc_lengths, mc_clip_lengths),
        ):
            if lengths.numel() == 0:
                continue
            metrics.update(
                {
                    f"{prefix}/mean": lengths.mean().item(),
                    f"{prefix}/max": lengths.max().item(),
                    f"{prefix}/min": lengths.min().item(),
                    f"{prefix}/clip_ratio": (lengths == clip_lengths).float().mean().item(),
                }
            )
        return metrics

    @staticmethod
    def _select_dynamic_filter_prompt_uids(
        prompt_uid2metric_vals: dict[object, list],
        *,
        branch_eligible_prompt_uids: set[object] | None,
        fill_shortfall: bool,
        prompts_needed: int,
    ) -> tuple[list[object], int]:
        """Select passing prompts and optionally fill the shortfall in input order."""

        eligible_prompt_uids = [
            uid
            for uid in prompt_uid2metric_vals
            if branch_eligible_prompt_uids is None or uid in branch_eligible_prompt_uids
        ]
        passed_prompt_uids = [
            uid
            for uid in eligible_prompt_uids
            if np.std(prompt_uid2metric_vals[uid]) > 0 or len(prompt_uid2metric_vals[uid]) == 1
        ]
        selected_prompt_uids = list(passed_prompt_uids)
        if fill_shortfall and len(selected_prompt_uids) < prompts_needed:
            passed_set = set(passed_prompt_uids)
            filtered_prompt_uids = [uid for uid in eligible_prompt_uids if uid not in passed_set]
            selected_prompt_uids.extend(filtered_prompt_uids[: prompts_needed - len(selected_prompt_uids)])
        return selected_prompt_uids, len(passed_prompt_uids)

    @staticmethod
    def _group_metric_values(batch: DataProto, metric_name: str) -> dict[object, list]:
        """Collect one dynamic-sampling metric list per prompt, preserving input order."""

        grouped = defaultdict(list)
        for uid, metric_val in zip(
            batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name], strict=True
        ):
            grouped[uid].append(metric_val)
        return grouped

    @staticmethod
    def _metric_has_variance(metric_vals: list) -> bool:
        return len(metric_vals) == 1 or np.std(metric_vals) > 0

    @staticmethod
    def _select_fixed_size_mixed_group(
        row_indices: list[int],
        metric_vals: list,
        *,
        target_size: int,
        generator: np.random.Generator,
    ) -> list[int]:
        """Sample a fixed-size subset while guaranteeing nonzero metric variance."""

        if target_size < 2:
            raise ValueError("A mixed dynamic-sampling group requires target_size >= 2.")
        if len(row_indices) != len(metric_vals) or len(row_indices) < target_size:
            raise ValueError("Adaptive top-up candidates do not match the requested group size.")
        if not RayDAPOTrainer._metric_has_variance(metric_vals):
            raise ValueError("Cannot construct a mixed group from zero-variance candidates.")

        values = np.asarray(metric_vals)
        first = int(generator.integers(len(row_indices)))
        different = np.flatnonzero(values != values[first])
        second = int(different[int(generator.integers(len(different)))])
        mandatory = {first, second}
        remaining = np.asarray([idx for idx in range(len(row_indices)) if idx not in mandatory], dtype=np.int64)
        extra_count = target_size - 2
        extras = generator.choice(remaining, size=extra_count, replace=False).tolist() if extra_count else []
        chosen = [first, second, *extras]
        generator.shuffle(chosen)
        return [row_indices[idx] for idx in chosen]

    @staticmethod
    def _select_rescued_group_preserving_initial_rows(
        row_indices: list[int],
        metric_vals: list,
        initial_row_is_eligible: list[bool],
        *,
        target_size: int,
        required_eligible_rows: int,
        generator: np.random.Generator,
    ) -> list[int]:
        """Replace one initial row with an opposite-label top-up without losing anchor eligibility."""

        initial_values = np.asarray(metric_vals[:target_size])
        extra_values = np.asarray(metric_vals[target_size:])
        opposite_offsets = np.flatnonzero(extra_values != initial_values[0])
        if len(opposite_offsets) == 0:
            raise ValueError("A rescued group must contain an opposite-metric top-up sample.")
        opposite_position = target_size + int(
            opposite_offsets[int(generator.integers(len(opposite_offsets)))]
        )

        eligible_count = sum(initial_row_is_eligible)
        removable = [
            position
            for position, is_eligible in enumerate(initial_row_is_eligible)
            if eligible_count - int(is_eligible) >= required_eligible_rows
        ]
        if not removable:
            return RayDAPOTrainer._select_fixed_size_mixed_group(
                row_indices,
                metric_vals,
                target_size=target_size,
                generator=generator,
            )
        removed_position = removable[int(generator.integers(len(removable)))]
        chosen = [
            row_index
            for position, row_index in enumerate(row_indices[:target_size])
            if position != removed_position
        ]
        chosen.append(row_indices[opposite_position])
        generator.shuffle(chosen)
        return chosen

    @staticmethod
    def _ensure_filter_metric(batch: DataProto, metric_name: str) -> None:
        if metric_name == "seq_final_reward":
            batch.non_tensor_batch["seq_final_reward"] = batch.batch["token_level_rewards"].sum(dim=-1).numpy()
        elif metric_name == "seq_reward":
            batch.non_tensor_batch["seq_reward"] = batch.batch["token_level_scores"].sum(dim=-1).numpy()

    def _attach_rollout_rewards(self, batch: DataProto, metrics: dict, timing_raw: dict) -> tuple[DataProto, dict]:
        """Score a generated rollout batch exactly as in the main DAPO path."""

        if self.use_rm and "rm_scores" not in batch.batch.keys():
            batch = batch.union(self._compute_reward_colocate(batch))

        reward_tensor, reward_extra_infos_dict = extract_reward(batch)
        batch.batch["token_level_scores"] = reward_tensor
        if reward_extra_infos_dict:
            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(
                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        return batch, reward_extra_infos_dict

    def _generate_adaptive_topup(
        self,
        *,
        prompt_context: DataProto,
        generation_context: DataProto,
        prompt_indices: list[int],
        samples_per_prompt: int,
        metric_name: str,
        metrics: dict,
        timing_raw: dict,
        attach_step_inputs: bool,
    ) -> DataProto:
        """Generate and score one top-up chunk for the pending prompts."""

        selected_generation_context = generation_context[prompt_indices]
        topup_input = selected_generation_context.repeat(repeat_times=samples_per_prompt, interleave=True)
        size_divisor = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
        padded_input, pad_size = pad_dataproto_to_divisor(topup_input, size_divisor)
        padded_output = self.async_rollout_manager.generate_sequences(padded_input)
        padded_output.meta_info.pop("timing", None)
        topup_output = unpad_dataproto(padded_output, pad_size=pad_size)

        topup_batch = prompt_context[prompt_indices].repeat(
            repeat_times=samples_per_prompt,
            interleave=True,
        )
        topup_batch = topup_batch.union(topup_output)
        if self.config.algorithm.use_kl_in_reward:
            topup_batch = self.compute_kl_related_metrics(topup_batch, metrics, timing_raw)
        topup_batch, _ = self._attach_rollout_rewards(topup_batch, metrics, timing_raw)
        self._ensure_filter_metric(topup_batch, metric_name)
        if attach_step_inputs:
            self._prepare_step_inputs(topup_batch)
        return topup_batch

    def _adaptive_topup_dynamic_filter_groups(
        self,
        batch: DataProto,
        *,
        prompt_context: DataProto,
        generation_context: DataProto,
        metric_name: str,
        metrics: dict,
        timing_raw: dict,
        topup_eligible_prompt_uids: set[object] | None = None,
        initial_anchor_eligible_rows: np.ndarray | None = None,
        required_anchor_rows: int = 0,
    ) -> DataProto:
        """Top up only initially homogeneous groups, then restore the original group size."""

        filter_config = self.config.algorithm.filter_groups
        topup_config = filter_config.get("adaptive_topup", None)
        if topup_config is None or not bool(topup_config.get("enabled", False)):
            return batch

        initial_n = int(self.config.actor_rollout_ref.rollout.n)
        max_total_n = int(topup_config.max_total_n)
        chunk_size = int(topup_config.chunk_size)
        initial_metrics = self._group_metric_values(batch, metric_name)
        prompt_order = prompt_context.non_tensor_batch["uid"].tolist()
        prompt_index = {uid: idx for idx, uid in enumerate(prompt_order)}
        pending = [
            uid
            for uid in prompt_order
            if not self._metric_has_variance(initial_metrics[uid])
            and (topup_eligible_prompt_uids is None or uid in topup_eligible_prompt_uids)
        ]
        if not pending:
            return batch

        candidate_batches = [batch]
        accumulated_metrics = {uid: list(vals) for uid, vals in initial_metrics.items()}
        generated_per_pending_prompt = 0
        while pending and initial_n + generated_per_pending_prompt < max_total_n:
            current_chunk_size = min(chunk_size, max_total_n - initial_n - generated_per_pending_prompt)
            topup_batch = self._generate_adaptive_topup(
                prompt_context=prompt_context,
                generation_context=generation_context,
                prompt_indices=[prompt_index[uid] for uid in pending],
                samples_per_prompt=current_chunk_size,
                metric_name=metric_name,
                metrics=metrics,
                timing_raw=timing_raw,
                attach_step_inputs="step_end_mask" in batch.batch,
            )
            # DataProto.concat requires matching metadata; top-up rollout timing
            # has already been consumed and is intentionally not attached.
            topup_batch.meta_info = batch.meta_info
            candidate_batches.append(topup_batch)
            topup_metrics = self._group_metric_values(topup_batch, metric_name)
            for uid in pending:
                accumulated_metrics[uid].extend(topup_metrics[uid])
            pending = [uid for uid in pending if not self._metric_has_variance(accumulated_metrics[uid])]
            generated_per_pending_prompt += current_chunk_size

        combined = DataProto.concat(candidate_batches)
        rows_by_prompt: dict[object, list[int]] = defaultdict(list)
        metrics_by_prompt: dict[object, list] = defaultdict(list)
        for row_index, (uid, metric_val) in enumerate(
            zip(combined.non_tensor_batch["uid"], combined.non_tensor_batch[metric_name], strict=True)
        ):
            rows_by_prompt[uid].append(row_index)
            metrics_by_prompt[uid].append(metric_val)

        generator = np.random.default_rng(
            int(topup_config.get("selection_seed", 42)) + int(self.global_steps) * 1_000_003
        )
        selected_rows = []
        for uid in prompt_order:
            # Already-passing and still-failing groups retain their untouched
            # initial n samples. Only successfully rescued groups are rebuilt.
            if self._metric_has_variance(initial_metrics[uid]) or not self._metric_has_variance(
                metrics_by_prompt[uid]
            ):
                selected_rows.extend(rows_by_prompt[uid][:initial_n])
                continue
            if initial_anchor_eligible_rows is not None:
                base_rows = rows_by_prompt[uid][:initial_n]
                selected_rows.extend(
                    self._select_rescued_group_preserving_initial_rows(
                        rows_by_prompt[uid],
                        metrics_by_prompt[uid],
                        initial_anchor_eligible_rows[base_rows].tolist(),
                        target_size=initial_n,
                        required_eligible_rows=required_anchor_rows,
                        generator=generator,
                    )
                )
            else:
                selected_rows.extend(
                    self._select_fixed_size_mixed_group(
                        rows_by_prompt[uid],
                        metrics_by_prompt[uid],
                        target_size=initial_n,
                        generator=generator,
                    )
                )
        return combined[selected_rows]

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)
        self._prepare_step_inputs(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            batch.batch["entropys"] = entropys.detach()
            response_masks = batch.batch["response_mask"]
            actor_config = self.config.actor_rollout_ref.actor
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=actor_config.loss_agg_mode,
                loss_scale_factor=actor_config.loss_scale_factor,
            )
            old_log_prob_metrics = {
                "actor/entropy": entropy_agg.detach().item(),
                "perf/mfu/actor_infer": old_log_prob_mfu,
            }
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def _append_mc_training_branches(
        self,
        batch: DataProto,
        branches: DataProto,
        timing_raw: dict,
    ) -> DataProto:
        """Prepare suffix-only MC branches and append them to the actor batch."""

        branches.meta_info["global_token_num"] = torch.sum(branches.batch["attention_mask"], dim=-1).tolist()
        self._prepare_step_inputs(branches)
        with marked_timer("mc_branch_old_log_prob", timing_raw, "blue"):
            old_log_prob, _ = self._compute_old_log_prob(branches)
            branch_entropys = old_log_prob.batch["entropys"]
            branches.batch["entropys"] = branch_entropys.detach()
            old_log_prob.batch.pop("entropys")
            branches = branches.union(old_log_prob)

        if self.use_reference_policy:
            with marked_timer("mc_branch_ref", timing_raw, "olive"):
                branches = branches.union(self._compute_ref_log_prob(branches))

        if "rollout_log_probs" in batch.batch:
            # Direct suffix generation currently returns tokens only.  The
            # actor has not changed since generation, so its recomputed
            # proximal log-probabilities are the matching behavior values.
            branches.batch["rollout_log_probs"] = branches.batch["old_log_probs"].clone()

        batch_size, response_width = batch.batch["responses"].shape
        batch.batch["credit_prefix_lengths"] = torch.zeros(batch_size, dtype=torch.long)
        batch.batch["is_mc_branch"] = torch.zeros(batch_size, dtype=torch.bool)
        branches.batch["credit_anchor_mask"] = torch.zeros((len(branches), response_width), dtype=torch.bool)
        branches.batch["credit_anchor_targets"] = torch.zeros((len(branches), response_width), dtype=torch.float32)
        branches.batch["credit_anchor_q_targets"] = torch.zeros((len(branches), response_width), dtype=torch.float32)
        branches.batch["credit_anchor_v_targets"] = torch.zeros((len(branches), response_width), dtype=torch.float32)
        if "global_token_num" in batch.meta_info:
            branches.meta_info["global_token_num"] = batch.meta_info["global_token_num"]

        base_keys = set(batch.batch.keys())
        branch_keys = set(branches.batch.keys())
        if base_keys != branch_keys:
            missing = sorted(base_keys - branch_keys)
            extra = sorted(branch_keys - base_keys)
            raise ValueError(
                f"MC training branches do not match the base actor batch schema; missing={missing}, extra={extra}."
            )
        return DataProto.concat([batch, branches])

    def _resume_counterfactual_credit_rollout(self) -> None:
        """Restore a sleeping hybrid rollout before auxiliary suffix generation."""

        checkpoint_backend = str(self.config.actor_rollout_ref.rollout.checkpoint_engine.backend).lower()
        if checkpoint_backend != "naive":
            raise ValueError(
                "Sparse counterfactual credit currently requires "
                "actor_rollout_ref.rollout.checkpoint_engine.backend=naive so a sleeping hybrid rollout "
                "can be restored without a second sleep cycle."
            )
        self.checkpoint_manager.update_weights(global_steps=self.global_steps)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self._data_epoch = 0
        self.max_steps_duration = 0

        credit_supervisor = None
        credit_config = self.config.algorithm.get("sparse_counterfactual_credit", None)
        if credit_config is not None and credit_config.get("enabled", False):
            if self.config.algorithm.adv_estimator not in (AdvantageEstimator.GRPO, AdvantageEstimator.GRPO.value):
                raise ValueError("Sparse counterfactual credit currently requires algorithm.adv_estimator=grpo")
            if self.use_rm:
                raise ValueError("Sparse counterfactual credit currently supports RLVR reward functions only")
            if self.config.algorithm.use_kl_in_reward:
                raise ValueError("Sparse counterfactual credit requires algorithm.use_kl_in_reward=False")
            actor_config = self.config.actor_rollout_ref.actor
            if credit_config.get("use_probe", True) and not actor_config.counterfactual_credit_head.enabled:
                raise ValueError("Sparse counterfactual credit requires actor.counterfactual_credit_head.enabled=True")
            if self.use_legacy_worker_impl == "disable":
                raise ValueError("Sparse counterfactual credit currently requires the legacy FSDP/FSDP2 worker API")
            if str(actor_config.strategy).lower() not in {"fsdp", "fsdp2"}:
                raise ValueError("Sparse counterfactual credit currently supports actor strategy fsdp or fsdp2")
            if int(actor_config.ulysses_sequence_parallel_size) != 1:
                raise ValueError("Sparse counterfactual credit requires ulysses_sequence_parallel_size=1")
            if actor_config.use_fused_kernels or actor_config.use_prefix_grouper:
                raise ValueError("Sparse counterfactual credit requires fused kernels and prefix grouping disabled")
            rollout_config = self.config.actor_rollout_ref.rollout
            if str(rollout_config.name).lower() != "vllm":
                raise ValueError("Sparse counterfactual credit currently requires actor_rollout_ref.rollout.name=vllm")
            if rollout_config.multi_turn.enable:
                raise ValueError("Sparse counterfactual credit currently supports single-turn rollouts only")
            if int(rollout_config.n) < 2:
                raise ValueError("Sparse counterfactual credit requires rollout.n >= 2")
            if credit_config.train_mc_branches:
                if not self.config.algorithm.filter_groups.enable:
                    raise ValueError("MC branch training requires algorithm.filter_groups.enable=True")
                inserted_branches = int(credit_config.branch_groups_per_prompt) * int(credit_config.num_samples)
                if inserted_branches != int(rollout_config.n):
                    raise ValueError(
                        "MC branch training requires branch_groups_per_prompt * num_samples == rollout.n "
                        "so each augmented prompt keeps equal base and branch halves"
                    )
                maximum_groups = 2 * int(credit_config.anchors_per_group)
                if int(credit_config.branch_groups_per_prompt) > maximum_groups:
                    raise ValueError("MC branch training requires branch_groups_per_prompt <= 2 * anchors_per_group")
            rollout_correction = self.config.algorithm.get("rollout_correction", None)
            if rollout_correction and rollout_correction.get("bypass_mode", False):
                raise ValueError("Sparse counterfactual credit requires recomputing actor old log probabilities")
            credit_supervisor = SparseCounterfactualCreditSupervisor(
                tokenizer=self.tokenizer,
                trainer_config=self.config,
                rollout_manager=self.async_rollout_manager,
                reward_loop_manager=self.reward_loop_manager,
            )
        filter_config = self.config.algorithm.get("filter_groups", None)
        if filter_config is not None:
            topup_config = filter_config.get("adaptive_topup", None)
            if topup_config is not None and bool(topup_config.get("enabled", False)):
                if not bool(filter_config.enable):
                    raise ValueError("Adaptive top-up requires algorithm.filter_groups.enable=True")
                initial_n = int(self.config.actor_rollout_ref.rollout.n)
                if int(topup_config.max_total_n) <= initial_n:
                    raise ValueError(
                        "algorithm.filter_groups.adaptive_topup.max_total_n must be greater than rollout.n"
                    )
        self.credit_supervisor = credit_supervisor

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_prompt_pass_filter = 0
        num_gen_batches = 0
        for epoch, batch_dict in self._iter_training_batches():
            if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
            metrics = {}

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )

            new_batch: DataProto = DataProto.from_single_dict(batch_dict)
            new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            num_gen_batches += 1
            gen_batch = self._get_gen_batch(new_batch)
            gen_batch_output = gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
            )

            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # generate a batch
                with marked_timer("gen", timing_raw, "red"):
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    with marked_timer("gen_max", timing_raw, "red"):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                        new_batch = new_batch.union(gen_baseline_output)
                        # compute reward model score on new_batch
                        rm_scores = None
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            rm_scores = self._compute_reward_colocate(new_batch)
                            new_batch = new_batch.union(rm_scores)
                        reward_baseline_tensor, _ = extract_reward(new_batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        keys_to_pop = set(gen_baseline_output.batch.keys())
                        if rm_scores is not None:
                            keys_to_pop.update(rm_scores.batch.keys())
                        new_batch.pop(batch_keys=list(keys_to_pop))

                        new_batch.batch["reward_baselines"] = reward_baseline_tensor

                        del rm_scores, gen_baseline_batch, gen_baseline_output

                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )
                prompt_context = new_batch
                generation_context = gen_batch
                generation_context.non_tensor_batch["uid"] = prompt_context.non_tensor_batch["uid"].copy()
                # repeat to align with repeated responses in rollout
                new_batch = prompt_context.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                new_batch = new_batch.union(gen_batch_output)

                if self.config.algorithm.use_kl_in_reward:
                    # We need these metrics for apply_kl_penalty if using kl in reward
                    new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                    # otherwise, we will compute those after dynamic sampling

                with marked_timer("reward", timing_raw, "yellow"):
                    new_batch, reward_extra_infos_dict = self._attach_rollout_rewards(
                        new_batch,
                        metrics,
                        timing_raw,
                    )

                if self.config.algorithm.filter_groups.enable:
                    metric_name = self.config.algorithm.filter_groups.metric
                    self._ensure_filter_metric(new_batch, metric_name)
                    topup_eligible_prompt_uids = None
                    initial_anchor_eligible_rows = None
                    required_anchor_rows = 0
                    if credit_supervisor is not None and bool(credit_config.train_mc_branches):
                        # Do not spend top-up rollouts on prompts that already
                        # fail the independently checkable MC-anchor constraint.
                        self._prepare_step_inputs(new_batch)
                        initial_anchor_eligible_rows = (
                            credit_supervisor.mc_branch_eligible_row_mask(new_batch).cpu().numpy()
                        )
                        required_anchor_rows = int(credit_config.anchors_per_group)
                        topup_eligible_prompt_uids = credit_supervisor.mc_branch_eligible_prompt_uids(new_batch)
                    new_batch = self._adaptive_topup_dynamic_filter_groups(
                        new_batch,
                        prompt_context=prompt_context,
                        generation_context=generation_context,
                        metric_name=metric_name,
                        metrics=metrics,
                        timing_raw=timing_raw,
                        topup_eligible_prompt_uids=topup_eligible_prompt_uids,
                        initial_anchor_eligible_rows=initial_anchor_eligible_rows,
                        required_anchor_rows=required_anchor_rows,
                    )

                branch_eligible_prompt_uids = None
                if credit_supervisor is not None and bool(credit_config.train_mc_branches):
                    # Step structure is known after the base rollout, before
                    # any expensive Q/V suffix sampling. Single-step
                    # responses cannot provide a trainable branch group.
                    self._prepare_step_inputs(new_batch)
                    branch_eligible_prompt_uids = credit_supervisor.mc_branch_eligible_prompt_uids(new_batch)

                if not self.config.algorithm.filter_groups.enable:
                    batch = new_batch
                else:  # NOTE: When prompts after filtering is less than train batch size,
                    # we skip to the next generation batch
                    metric_name = self.config.algorithm.filter_groups.metric
                    # Collect the sequence reward for each trajectory
                    prompt_uid2metric_vals = self._group_metric_values(new_batch, metric_name)

                    prompt_bsz = self.config.data.train_batch_size
                    fill_shortfall = bool(self.config.algorithm.filter_groups.get("fill_shortfall", False))
                    kept_prompt_uids, passed_in_batch = self._select_dynamic_filter_prompt_uids(
                        prompt_uid2metric_vals,
                        branch_eligible_prompt_uids=branch_eligible_prompt_uids,
                        fill_shortfall=fill_shortfall,
                        prompts_needed=max(prompt_bsz - num_prompt_in_batch, 0),
                    )
                    num_prompt_pass_filter += passed_in_batch
                    num_prompt_in_batch += len(kept_prompt_uids)

                    kept_traj_idxs = []
                    for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                        if traj_from_prompt_uid in kept_prompt_uids:
                            kept_traj_idxs.append(idx)

                    new_batch = new_batch[kept_traj_idxs]
                    batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                    if num_prompt_in_batch < prompt_bsz:
                        print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                        if fill_shortfall:
                            raise ValueError(
                                "algorithm.filter_groups.fill_shortfall=True could not fill one training batch "
                                f"without another rollout; selected={num_prompt_in_batch}, required={prompt_bsz}. "
                                "Increase data.gen_batch_size or relax the additional prompt eligibility "
                                "constraints."
                            )
                        max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                        if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                            print(f"{num_gen_batches=}. Keep generating...")
                            self.gen_steps += 1
                            is_last_step = self.global_steps >= self.total_training_steps
                            continue
                        else:
                            raise ValueError(
                                f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                + " Generated too many. Please check if your data are too difficult."
                                + " You could also try set max_num_gen_batches=0 to enable endless trials."
                            )
                    else:
                        # Align the batch
                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        batch = batch[:traj_bsz]

                self.checkpoint_manager.sleep_replicas()

                # === Updating ===
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # TODO: Decouple the DP balancing and mini-batching.
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if not self.config.algorithm.use_kl_in_reward:
                    batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                if credit_supervisor is not None:
                    credit_coef = credit_advantage_coefficient(
                        global_step=self.global_steps,
                        total_training_steps=self.total_training_steps,
                        maximum=float(credit_config.advantage_coef),
                        warmup_ratio=float(credit_config.warmup_ratio),
                    )
                    with marked_timer("counterfactual_credit_rollout", timing_raw, "purple"):
                        self._resume_counterfactual_credit_rollout()
                        try:
                            credit_rollout_result = credit_supervisor.collect_targets(
                                batch,
                                global_step=self.global_steps,
                            )
                            metrics.update(credit_rollout_result.metrics)
                        finally:
                            self.checkpoint_manager.sleep_replicas()
                    if credit_rollout_result.training_branches is not None:
                        batch = self._append_mc_training_branches(
                            batch,
                            credit_rollout_result.training_branches,
                            timing_raw,
                        )
                        if self.config.trainer.balance_batch:
                            self._balance_batch(
                                batch,
                                metrics=metrics,
                                logging_prefix="mc_augmented_global_seqlen",
                            )
                        batch.meta_info["global_token_num"] = torch.sum(
                            batch.batch["attention_mask"], dim=-1
                        ).tolist()
                    batch = self._predict_counterfactual_credit(batch, credit_coef, metrics, timing_raw)

                batch = self._prepare_auxiliary_rewards(batch, metrics, timing_raw)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, "cyan"):
                        values = self._compute_values(batch)
                        batch = batch.union(values)

                # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                    batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                    # IS and off-policy metrics already have rollout_corr/ prefix
                    metrics.update(is_metrics)

                with marked_timer("adv", timing_raw, "brown"):
                    # compute advantages, executed on the driver process
                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )
                    if credit_supervisor is not None:
                        endpoint_credit = merge_anchor_credit(
                            batch.batch["credit_predictions"],
                            batch.batch["credit_anchor_mask"],
                            batch.batch["credit_anchor_targets"][batch.batch["credit_anchor_mask"]],
                        )
                        credit_residual, credit_metrics = build_credit_residual(
                            endpoint_credit,
                            batch.batch["step_end_mask"],
                            batch.batch["response_mask"],
                            uids=batch.non_tensor_batch["uid"],
                            epsilon=float(credit_config.epsilon),
                        )
                        batch.batch["advantages"] = (
                            batch.batch["advantages"] + credit_coef * credit_residual
                        ) * batch.batch["response_mask"]
                        batch.batch["counterfactual_credit_residual"] = credit_residual
                        metrics.update(credit_metrics)
                        metrics["credit/advantage_coef"] = credit_coef
                    batch = self._augment_advantages(batch, metrics)

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, "pink"):
                        critic_output = self._update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, "red"):
                        actor_output = self._update_actor(batch)

                    # Check if ESI/training plan is close to expiration
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                    with marked_timer("update_weights", timing_raw, "red"):
                        self.checkpoint_manager.update_weights()
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    # Filtering, adaptive top-up, and MC branch insertion can
                    # all reorder the batch. Re-read reward extras from the
                    # final batch instead of retaining their pre-filter order.
                    reward_extra_infos_dict = {
                        key: batch.non_tensor_batch[key].tolist()
                        for key in reward_extra_infos_dict
                        if key in batch.non_tensor_batch
                    }
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

            # validate
            if self.config.trainer.test_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.test_freq == 0
            ):
                with marked_timer("testing", timing_raw, "green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

            steps_duration = timing_raw.get("step", 0)
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            if credit_supervisor is not None:
                metrics.update(
                    self._compute_branch_response_length_metrics(
                        batch,
                        mc_branch_max_response_length=credit_supervisor._counterfactual_response_limit(),
                    )
                )
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            timing_raw = defaultdict(float)  # clear timing

            metrics["train/num_gen_batches"] = num_gen_batches
            if self.config.algorithm.filter_groups.enable:
                # Count every prompt group that passed dynamic sampling
                # before the batch is truncated to data.train_batch_size.
                metrics["train/num_prompt_pass_filter"] = num_prompt_pass_filter
            batch = None
            num_prompt_in_batch = 0
            num_prompt_pass_filter = 0
            num_gen_batches = 0

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1
            self.gen_steps += 1
        # Dynamic sampling can exhaust the dataloader before a complete optimizer
        # step reaches the in-loop finalization path.
        timing_raw = defaultdict(float)
        final_metrics = {}

        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()

        if self.config.trainer.test_freq > 0:
            with marked_timer("testing", timing_raw, "green"):
                last_val_metrics = self._validate()
            final_metrics.update(last_val_metrics)

        final_metrics.update({f"timing_s/{key}": value for key, value in timing_raw.items()})
        if final_metrics:
            logger.log(data=final_metrics, step=self.global_steps)

        pprint(f"Final validation metrics: {last_val_metrics}")
        progress_bar.close()
