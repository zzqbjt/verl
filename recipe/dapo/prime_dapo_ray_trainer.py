# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DAPO with PRIME's online implicit process reward model."""

import os
from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.trainer.ppo.utils import Role
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer

from .dapo_ray_trainer import RayDAPOTrainer


def compute_prime_process_returns(
    token_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: np.ndarray,
    *,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Apply PRIME's response-group baseline and form token-level returns.

    PRIME first averages each response's implicit token rewards, uses those
    averages to construct a leave-one-out group baseline, and only then forms
    the discounted return.  Grouping by uid instead of row position keeps the
    calculation correct if a trainer later reorders the batch.
    """

    if token_rewards.ndim != 2 or response_mask.shape != token_rewards.shape:
        raise ValueError("token_rewards and response_mask must be matching rank-2 tensors")
    if len(group_ids) != token_rewards.shape[0]:
        raise ValueError("group_ids must contain one id per response")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    mask = response_mask.to(device=token_rewards.device, dtype=torch.bool)
    rewards = token_rewards.float() * mask
    valid_counts = mask.sum(dim=-1)
    if torch.any(valid_counts == 0):
        raise ValueError("PRIME requires every response to contain at least one valid token")
    response_means = rewards.sum(dim=-1) / valid_counts

    rows_by_group = defaultdict(list)
    for row, group_id in enumerate(group_ids):
        rows_by_group[group_id].append(row)

    centered_rewards = torch.zeros_like(rewards)
    for group_id, rows in rows_by_group.items():
        if len(rows) < 2:
            raise ValueError(f"PRIME requires at least two responses per prompt; group {group_id!r} has one")
        row_index = torch.as_tensor(rows, device=rewards.device, dtype=torch.long)
        group_size = len(rows)
        baseline_numerator = response_means[row_index].sum()
        centered_rewards[row_index] = (
            rewards[row_index] * (group_size / (group_size - 1)) - baseline_numerator / (group_size - 1)
        ) * mask[row_index]

    returns = torch.zeros_like(centered_rewards)
    running = torch.zeros(token_rewards.shape[0], device=token_rewards.device, dtype=centered_rewards.dtype)
    for position in reversed(range(token_rewards.shape[1])):
        running = centered_rewards[:, position] + gamma * running
        running = running * mask[:, position]
        returns[:, position] = running
    return returns


def _masked_stats(values: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    selected = values[mask.bool()].float()
    return selected.mean().item(), selected.std(unbiased=False).item()


class RayPrimeDAPOTrainer(RayDAPOTrainer):
    """Keep DAPO sampling and policy optimization, adding PRIME process returns."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prime_config = self.config.get("prime", None)
        if prime_config is None or not bool(prime_config.get("enabled", False)):
            raise ValueError("RayPrimeDAPOTrainer requires prime.enabled=True")
        if self.use_legacy_worker_impl == "disable":
            raise ValueError("DAPO + PRIME requires the legacy FSDP worker implementation")
        if str(self.config.actor_rollout_ref.actor.strategy).lower() not in {"fsdp", "fsdp2"}:
            raise ValueError("DAPO + PRIME requires actor.strategy=fsdp or fsdp2")
        if self.config.trainer.balance_batch:
            raise ValueError("DAPO + PRIME currently requires trainer.balance_batch=False")
        sparse_credit = self.config.algorithm.get("sparse_counterfactual_credit", {})
        if bool(sparse_credit.get("enabled", False)):
            raise ValueError("DAPO + PRIME must not be combined with sparse counterfactual credit")
        self.config.prime.model.optim.total_training_steps = self.total_training_steps

    def _register_additional_worker_roles(self) -> None:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        reward_model_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.RewardModel],
            config=self.config.prime,
        )
        self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = reward_model_cls

    def _initialize_additional_worker_groups(self, all_wg: dict) -> None:
        self.prime_rm_wg = all_wg[str(Role.RewardModel)]
        self.prime_rm_wg.init_model()

    def _prepare_auxiliary_rewards(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
    ) -> DataProto:
        prime_config = self.config.prime
        correctness_key = str(prime_config.get("correctness_key", "acc"))
        if correctness_key in batch.batch:
            correctness = batch.batch[correctness_key].float()
        elif correctness_key in batch.non_tensor_batch:
            correctness = torch.as_tensor(
                np.asarray(batch.non_tensor_batch[correctness_key], dtype=np.float32),
                dtype=torch.float32,
            )
        else:
            raise KeyError(f"PRIME correctness label {correctness_key!r} is missing from the rollout batch")
        if correctness.shape != (len(batch),):
            raise ValueError("PRIME correctness labels must contain one scalar per response")

        # The upstream PRIME worker expects the conventional name `acc`.
        batch.batch["acc"] = correctness
        batch.meta_info["n"] = int(self.config.actor_rollout_ref.rollout.n)

        update_style = str(prime_config.get("update", "before")).lower()
        with marked_timer("prime_reward_model", timing_raw, "purple"):
            if update_style == "before":
                # Keep both phases on the same worker with a batch-local frozen-ref cache.
                batch.meta_info["prime_score_after_update"] = True
                try:
                    score_output = self.prime_rm_wg.update_rm(batch)
                finally:
                    batch.meta_info.pop("prime_score_after_update", None)
                self._merge_prime_metrics(score_output, metrics)
            elif update_style == "after":
                # update_rm returns rewards from the forward pass immediately
                # preceding its optimizer step, which is PRIME's single-forward mode.
                score_output = self.prime_rm_wg.update_rm(batch)
                self._merge_prime_metrics(score_output, metrics)
            elif update_style == "none":
                score_output = self.prime_rm_wg.compute_rm_score(batch)
                self._merge_prime_metrics(score_output, metrics)
            else:
                raise ValueError("prime.update must be one of: before, after, none")

        batch.batch["prime_rm_scores"] = score_output.batch["rm_scores"]
        return batch

    @staticmethod
    def _merge_prime_metrics(output: DataProto, metrics: dict) -> None:
        worker_metrics = output.meta_info.get("metrics", None)
        if worker_metrics:
            metrics.update(reduce_metrics(worker_metrics))

    def _augment_advantages(self, batch: DataProto, metrics: dict) -> DataProto:
        if "prime_rm_scores" not in batch.batch:
            raise ValueError("PRIME rewards must be computed before advantage estimation")

        prime_config = self.config.prime
        response_mask = batch.batch["response_mask"]
        outcome_advantages = batch.batch["advantages"].float() * response_mask
        process_returns = compute_prime_process_returns(
            batch.batch["prime_rm_scores"],
            response_mask,
            batch.non_tensor_batch["uid"],
            gamma=float(prime_config.get("gamma", 1.0)),
        )

        outcome_coef = float(prime_config.get("outcome_advantage_coef", 1.0))
        process_coef = float(prime_config.get("process_advantage_coef", 1.0))
        combined = outcome_coef * outcome_advantages + process_coef * process_returns
        if bool(prime_config.get("normalize_advantage", True)):
            combined = verl_F.masked_whiten(combined, response_mask)
        combined = combined * response_mask

        outcome_mean, outcome_std = _masked_stats(outcome_advantages, response_mask)
        process_mean, process_std = _masked_stats(process_returns, response_mask)
        combined_mean, combined_std = _masked_stats(combined, response_mask)
        metrics.update(
            {
                "prime/outcome_advantage_mean": outcome_mean,
                "prime/outcome_advantage_std": outcome_std,
                "prime/process_return_mean": process_mean,
                "prime/process_return_std": process_std,
                "prime/combined_advantage_mean": combined_mean,
                "prime/combined_advantage_std": combined_std,
                "prime/outcome_advantage_coef": outcome_coef,
                "prime/process_advantage_coef": process_coef,
            }
        )
        batch.batch["advantages"] = combined
        batch.batch["returns"] = combined
        return batch

    def _save_trainer_extra_state(self, local_global_step_folder: str) -> None:
        super()._save_trainer_extra_state(local_global_step_folder)
        self.prime_rm_wg.save_checkpoint(
            os.path.join(local_global_step_folder, "prime_reward_model"),
            global_step=self.global_steps,
        )

    def _load_trainer_extra_state(self, global_step_folder: str) -> None:
        super()._load_trainer_extra_state(global_step_folder)
        reward_path = os.path.join(global_step_folder, "prime_reward_model")
        if not os.path.isdir(reward_path):
            raise FileNotFoundError(f"PRIME reward-model checkpoint is missing: {reward_path}")
        self.prime_rm_wg.load_checkpoint(
            reward_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )
