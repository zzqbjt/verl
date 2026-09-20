# Copyright 2024 PRIME team and/or its affiliates
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
Implement a multiprocess PPOCritic
"""

import torch
import torch.distributed
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches, restore_dynamic_batch
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

from .prime_core_algos import compute_ce_dpo_loss_rm, compute_detach_dpo_loss_rm

__all__ = ["DataParallelPRIMERewardModel"]


class DataParallelPRIMERewardModel:
    def __init__(self, config, reward_module: nn.Module, ref_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Reward model use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        print(f"Reward model use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)

    def _model_log_probs(self, model, micro_batch, prompt_length):
        """Compute only sampled-token log probabilities, with identical packing for RM/ref."""
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        num_actions = seqlen - prompt_length
        if self.use_remove_padding:
            packed, indices, *_ = unpad_input(input_ids.unsqueeze(-1), micro_batch["attention_mask"])
            packed = packed.transpose(0, 1)
            positions = index_first_axis(
                rearrange(micro_batch["position_ids"].unsqueeze(-1), "b s ... -> (b s) ..."), indices
            ).transpose(0, 1)
            labels = torch.roll(packed, shifts=-1, dims=1)
            if self.ulysses_sequence_parallel_size > 1:
                packed, positions, pad_size = ulysses_pad_and_slice_inputs(
                    packed, positions, sp_size=self.ulysses_sequence_parallel_size
                )
                labels, _, _ = ulysses_pad_and_slice_inputs(
                    labels, None, self.ulysses_sequence_parallel_size
                )
            output = model(
                input_ids=packed, attention_mask=None, position_ids=positions,
                use_cache=False, return_dict=True,
            )
            if self.use_fused_kernels:
                log_probs = output.log_probs.squeeze(0).float()
            else:
                log_probs = verl_F.logprobs_from_logits(output.logits.squeeze(0), labels.squeeze(0))
            if self.ulysses_sequence_parallel_size > 1:
                log_probs = gather_outputs_and_unpad(
                    log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )
            log_probs = pad_input(
                log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            ).squeeze(-1)
            return log_probs[:, -num_actions - 1 : -1]
        output = model(
            input_ids=input_ids, attention_mask=micro_batch["attention_mask"],
            position_ids=micro_batch["position_ids"], use_cache=False, return_dict=True,
        )
        if self.use_fused_kernels:
            return output.log_probs[:, -num_actions - 1 : -1].float()
        return verl_F.logprobs_from_logits(
            output.logits[:, -num_actions - 1 : -1], input_ids[:, -num_actions:]
        )

    @torch.no_grad()
    def cache_reference_log_probs(self, data: DataProto):
        """Batch-local cache, refreshed for each update and reused for post-update scoring."""
        if self.ref_module is None:
            return
        self.ref_module.eval()
        batch = data.select(batch_keys=["input_ids", "attention_mask", "position_ids"]).batch
        prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]
        if self.config.use_dynamic_bsz:
            micro_batches, indices = rearrange_micro_batches(
                batch=batch,
                max_token_len=self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size,
                dp_group=torch.distributed.group.WORLD if torch.distributed.is_initialized() else None,
            )
        else:
            micro_batches, indices = batch.split(self.config.micro_batch_size_per_gpu), None
        results = []
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            for micro_batch in micro_batches:
                results.append(self._model_log_probs(self.ref_module, micro_batch, prompt_length))
        log_probs = torch.cat(results, dim=0)
        if indices is not None:
            log_probs = restore_dynamic_batch(log_probs, indices)
        if data.batch.is_locked:
            # Worker inputs can be locked. Copy only the container so adding the
            # cache leaves the original lock/keys intact and shares input tensors.
            data.batch = data.batch.clone(recurse=False)
        data.batch["prime_ref_log_probs"] = log_probs.detach()

    def _forward_micro_batch(self, micro_batch, prompt_length):
        num_actions = micro_batch["input_ids"].shape[-1] - prompt_length
        max_positions = micro_batch["attention_mask"][:, prompt_length:].sum(-1)
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            rm_log_labels = self._model_log_probs(self.reward_module, micro_batch, prompt_length)
        if "prime_ref_log_probs" in micro_batch:
            ref_log_labels = micro_batch["prime_ref_log_probs"]
        elif self.ref_module is not None:
            with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                ref_log_labels = self._model_log_probs(self.ref_module, micro_batch, prompt_length)
        else:
            ref_log_labels = micro_batch["old_log_probs"]
        ref_log_labels = ref_log_labels.to(rm_log_labels.dtype)
        q = rm_log_labels[:, -num_actions:] - ref_log_labels[:, -num_actions:]  # this is actually diff of q

        # trim unnecessary logprobs here
        for i in range(micro_batch["input_ids"].shape[0]):
            q[i, max_positions[i] :] = 0

        # reward computation does not need gradient. only q needs
        with torch.no_grad():
            # generalized estimation of r should go before the reward filling. r means process reward for policy
            # model, or the advantage of reward model.
            lam = self.config.get("lambda", 0.0)
            beta = self.config.model.get("beta_train", 0.05)
            if lam == 0.0:
                r = q * beta
            else:
                # reward coefficient takes no effect here
                acc = micro_batch["acc"]
                q_ = q * beta
                r = torch.zeros_like(q)
                lastgaelam = 0
                # change the last token and mask out all paddings to make this process easier if we rely on
                # outcome reward to calculate V
                for i in range(q.shape[0]):
                    if self.config.prime_use_gt:
                        q_[i, max_positions[i] - 1] = acc[i] - q_[i, : max_positions[i] - 1].sum()
                    q_[i, max_positions[i] :] = 0

                for t in reversed(range(num_actions)):
                    delta = q_[:, t]
                    lastgaelam = delta + lam * lastgaelam
                    r[:, t] = lastgaelam

            token_level_score = torch.zeros_like(q)

            if self.config.prime_granularity == "token":
                for i in range(micro_batch["input_ids"].shape[0]):
                    token_level_score[i, : max_positions[i] - 1] = r[i, : max_positions[i] - 1]
            elif self.config.prime_granularity == "whole":
                for i in range(micro_batch["input_ids"].shape[0]):
                    token_level_score[i, max_positions[i] - 1] = r[i, : max_positions[i]]
            else:
                raise NotImplementedError

        return token_level_score, q

    def _optimizer_step(self):
        assert self.config.model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip
            )
        self.reward_optimizer.step()
        return grad_norm

    def prime_norm(self, token_level_scores):
        if self.config.prime_norm == "batch_norm":
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
        return token_level_scores

    def compute_rm_score(self, data: DataProto):
        self.reward_module.eval()
        if self.ref_module is not None:
            self.ref_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "acc"]
        select_keys += [key for key in ("prime_ref_log_probs", "old_log_probs") if key in data.batch]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(
                batch=batch, max_token_len=max_token_len,
                dp_group=torch.distributed.group.WORLD if torch.distributed.is_initialized() else None,
            )
        else:
            micro_batches = batch.split(micro_batch_size)

        rm_scores_lst = []
        q_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                rm_score, q = self._forward_micro_batch(micro_batch, prompt_length)
            rm_scores_lst.append(rm_score)
            q_lst.append(q)
        rm_scores = torch.concat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        if use_dynamic_bsz:
            rm_scores = restore_dynamic_batch(rm_scores, indices)
            q = restore_dynamic_batch(q, indices)

        return (
            rm_scores,
            q.detach(),
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            },
        )

    def update_rm(self, data: DataProto):
        # make sure we are in training mode
        self.reward_module.train()
        if self.ref_module is not None:
            self.ref_module.eval()
        metrics = {}

        beta = self.config.model.get("beta_train", 0.05)

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts"]

        for key in ["Q_bc", "acc_bc", "prime_ref_log_probs", "old_log_probs"]:
            if key in data.batch.keys():
                select_keys.append(key)

        batch = data.select(batch_keys=select_keys).batch
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst = []
        q_lst = []

        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch, max_token_len=max_token_len,
                    dp_group=torch.distributed.group.WORLD if torch.distributed.is_initialized() else None,
                )
            else:
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                indices = None

            self.reward_optimizer.zero_grad()
            mini_scores, mini_q = [], []

            for data in micro_batches:
                data = data.to(get_device_name())
                attention_mask = data["attention_mask"]
                acc = data["acc"]

                prompt_ids = data["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_mask = attention_mask[:, prompt_length:]

                rm_score, q = self._forward_micro_batch(data, prompt_length)

                mini_scores.append(rm_score.detach())
                mini_q.append(q.detach())

                if self.config.model.loss_type == "ce":
                    dpo_loss = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
                elif self.config.model.loss_type == "dpo":
                    # the implementation of dpo is actually detached, which means we have to know the average
                    # value of w/l reward before the update.
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q, acc, Q_bc=data["Q_bc"], acc_bc=data["acc_bc"], response_mask=response_mask, beta=beta
                    )
                elif self.config.model.loss_type == "bon_acc":
                    # change the original distribution of each sample to BoN distribution, then update reward model
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=data["Q_bc"],
                        acc_bc=data["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_acc",
                    )
                elif self.config.model.loss_type == "bon_rm":
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=data["Q_bc"],
                        acc_bc=data["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_rm",
                    )
                else:
                    raise NotImplementedError

                # Both CE and detached DPO losses average over responses.
                # Weight by actual micro-batch size, including partial batches.
                loss = dpo_loss * (len(data) / len(mini_batch))

                loss.backward()

                append_to_dict(metrics, {"reward_model/dpo_loss": dpo_loss.detach().item()})

            scores = torch.cat(mini_scores, dim=0)
            q_values = torch.cat(mini_q, dim=0)
            if indices is not None:
                scores = restore_dynamic_batch(scores, indices)
                q_values = restore_dynamic_batch(q_values, indices)
            rm_scores_lst.append(scores)
            q_lst.append(q_values)
            grad_norm = self._optimizer_step()
            data = {"reward_model/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.reward_optimizer.zero_grad()

        rm_scores = torch.cat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        metrics.update(
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            }
        )

        return rm_scores, metrics
