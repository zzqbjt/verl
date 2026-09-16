# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Isolated FSDP actor for SPO; ordinary DAPO actors are unchanged."""

from collections import defaultdict

import torch.distributed as dist

from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor

from .spo_tree_core import SPOTreeConfig, spo_policy_loss, whiten_advantages


class SPODataParallelPPOActor(DataParallelPPOActor):
    spo_config: SPOTreeConfig

    def update_policy(self, data: DataProto):
        self.actor_module.train()
        config = self.spo_config
        temperature = float(data.meta_info["temperature"])
        pad_token_id = int(data.meta_info["pad_token_id"])
        keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "spo_real_rows",
        ]
        data = data.select(batch_keys=keys, non_tensor_batch_keys=[])
        if len(data) % self.config.ppo_mini_batch_size:
            raise ValueError("SPO actor batch must contain complete prompt mini-batches")
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        metrics = defaultdict(float)
        grad_norms = []
        distributed = dist.is_initialized()
        dp_group = self._get_micro_batch_sync_group() if distributed else None
        update_count = len(mini_batches) * self.config.ppo_epochs
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                real_count = int(mini_batch.batch["spo_real_rows"].sum())
                if self.config.use_dynamic_bsz:
                    # Keep dummy slots until after partitioning, so all ranks
                    # can participate in the same number of FSDP collectives.
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch,
                        max_token_len=self.config.ppo_max_token_len_per_gpu,
                        dp_group=dp_group,
                    )
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                self.actor_optimizer.zero_grad()
                for micro_batch in micro_batches:
                    real_mask = micro_batch.batch["spo_real_rows"].bool()
                    micro_real_count = int(real_mask.sum())
                    # Remove layout padding before whitening and
                    # policy loss. An all-dummy rank still executes one zero-loss
                    # forward/backward to match the other ranks' collectives.
                    micro_batch = micro_batch[real_mask] if micro_real_count else micro_batch[:1]
                    micro_batch = micro_batch.to(get_device_id())
                    inputs = {**micro_batch.batch, "pad_token_id": pad_token_id}
                    advantages, action_mask = whiten_advantages(
                        inputs["advantages"],
                        inputs["response_mask"],
                        inputs["old_log_probs"],
                        threshold=config.probability_threshold,
                        epsilon=config.whiten_epsilon,
                        distributed=distributed,
                        group=dp_group,
                    )
                    outputs = self._forward_micro_batch(inputs, temperature=temperature, calculate_entropy=False)
                    # Always retain frozen pre-update probabilities. Do not use
                    # the ordinary actor's single-minibatch on-policy shortcut.
                    loss, step_metrics = spo_policy_loss(
                        outputs["log_probs"],
                        inputs["old_log_probs"],
                        advantages,
                        action_mask,
                        actor_config=self.config,
                    )
                    scale = micro_real_count / max(real_count, 1)
                    scaled_loss = loss * scale
                    if self.scaler is None:
                        scaled_loss.backward()
                    else:
                        self.scaler.scale(scaled_loss).backward()
                    for key, value in step_metrics.items():
                        metrics[key] += float(value) * scale / update_count
                    segment_tokens = inputs["response_mask"].sum().clamp_min(1)
                    metrics["spo/probability_mask_keep_ratio"] += (
                        float(action_mask.sum() / segment_tokens) * scale / update_count
                    )
                grad_norms.append(float(self._optimizer_step()))
        self.actor_optimizer.zero_grad()
        metrics["actor/grad_norm"] = sum(grad_norms) / max(len(grad_norms), 1)
        metrics["spo/optimizer_steps"] = len(grad_norms)
        return dict(metrics)
