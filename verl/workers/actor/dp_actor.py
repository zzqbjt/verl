# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os
import re
from bisect import bisect_right
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
        answer_teacher_module (nn.Module, optional): Independent FSDP teacher used for answer scoring.
    """

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        answer_teacher_module: nn.Module | None = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.answer_teacher_module = answer_teacher_module
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

        self.answer_log_prob_ema_alpha = float(self.config.get("answer_log_prob_ema_alpha", 1.0))
        if not 0.0 <= self.answer_log_prob_ema_alpha <= 1.0:
            raise ValueError(
                "answer_log_prob_ema_alpha must be in [0, 1], "
                f"got {self.answer_log_prob_ema_alpha}."
            )
        if (
            self.actor_optimizer is not None
            and self.answer_log_prob_ema_alpha < 1.0
            and self.answer_teacher_module is None
        ):
            raise ValueError("An independent answer_teacher_module is required when answer_log_prob_ema_alpha < 1.")
        if self.answer_teacher_module is not None:
            self.answer_teacher_module.eval()
            self._copy_actor_to_answer_teacher()

        self._delimiter_token_id_cache = {}

    @staticmethod
    def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Return the tensor shard owned by this rank for a parameter or buffer."""
        if isinstance(tensor, DTensor):
            return tensor._local_tensor
        return tensor

    @torch.no_grad()
    def _copy_named_tensors(
        self,
        source_tensors: dict[str, torch.Tensor],
        target_tensors: dict[str, torch.Tensor],
    ) -> None:
        if source_tensors.keys() != target_tensors.keys():
            raise RuntimeError("Actor and answer teacher tensor names do not match.")
        for name, source_tensor in source_tensors.items():
            source_local_tensor = self._local_tensor(source_tensor).detach()
            target_local_tensor = self._local_tensor(target_tensors[name])
            if source_local_tensor.shape != target_local_tensor.shape:
                raise RuntimeError(
                    f"Actor tensor {name!r} has shape {source_local_tensor.shape}, "
                    f"but the answer teacher shard has shape {target_local_tensor.shape}."
                )
            target_local_tensor.copy_(
                source_local_tensor.to(device=target_local_tensor.device, dtype=target_local_tensor.dtype)
            )

    @torch.no_grad()
    def _copy_actor_to_answer_teacher(self) -> None:
        """Reset the independent teacher to exactly match the online actor."""
        if self.answer_teacher_module is None:
            return
        self._copy_named_tensors(
            dict(self.actor_module.named_parameters()),
            dict(self.answer_teacher_module.named_parameters()),
        )
        self._copy_named_tensors(
            dict(self.actor_module.named_buffers()),
            dict(self.answer_teacher_module.named_buffers()),
        )

    @torch.no_grad()
    def update_answer_teacher(self) -> None:
        """Apply ``teacher <- (1 - alpha) * teacher + alpha * actor`` once."""
        if self.answer_teacher_module is None:
            return

        alpha = self.answer_log_prob_ema_alpha
        if alpha == 0.0:
            return
        actor_parameters = dict(self.actor_module.named_parameters())
        teacher_parameters = dict(self.answer_teacher_module.named_parameters())
        if actor_parameters.keys() != teacher_parameters.keys():
            raise RuntimeError("Actor and answer teacher parameter names do not match.")

        for name, actor_parameter in actor_parameters.items():
            if not actor_parameter.requires_grad:
                continue
            actor_local_parameter = self._local_tensor(actor_parameter).detach()
            teacher_local_parameter = self._local_tensor(teacher_parameters[name])
            actor_local_parameter = actor_local_parameter.to(
                device=teacher_local_parameter.device,
                dtype=teacher_local_parameter.dtype,
            )
            if teacher_local_parameter.is_floating_point() or teacher_local_parameter.is_complex():
                teacher_local_parameter.lerp_(actor_local_parameter, alpha)
            else:
                teacher_local_parameter.copy_(actor_local_parameter)

        # Buffers are not part of theta, but copying them keeps model-specific
        # inference state (for example QAT observers) coherent with the actor.
        self._copy_named_tensors(
            dict(self.actor_module.named_buffers()),
            dict(self.answer_teacher_module.named_buffers()),
        )

    def answer_teacher_state_dict(self) -> dict | None:
        """Return the rank-local EMA teacher state for checkpointing."""
        if self.answer_teacher_module is None:
            return None
        return {
            "ema_alpha": self.answer_log_prob_ema_alpha,
            "parameters": {
                name: self._local_tensor(parameter).detach().cpu()
                for name, parameter in self.answer_teacher_module.named_parameters()
            },
            "buffers": {
                name: self._local_tensor(buffer).detach().cpu()
                for name, buffer in self.answer_teacher_module.named_buffers()
            },
        }

    @property
    def has_answer_teacher(self) -> bool:
        return self.answer_teacher_module is not None

    @torch.no_grad()
    def load_answer_teacher_state_dict(self, state_dict: dict | None) -> None:
        """Restore the rank-local EMA teacher, or reset it from the actor for old checkpoints."""
        if self.answer_teacher_module is None:
            return
        if state_dict is None:
            self._copy_actor_to_answer_teacher()
            return

        saved_parameters = state_dict.get("parameters", state_dict)
        teacher_parameters = dict(self.answer_teacher_module.named_parameters())
        if saved_parameters.keys() != teacher_parameters.keys():
            raise RuntimeError("EMA teacher checkpoint parameters do not match the actor parameters.")
        for name, saved_parameter in saved_parameters.items():
            teacher_parameter = self._local_tensor(teacher_parameters[name])
            if saved_parameter.shape != teacher_parameter.shape:
                raise RuntimeError(
                    f"EMA teacher checkpoint parameter {name!r} has shape {saved_parameter.shape}, "
                    f"but the actor shard has shape {teacher_parameter.shape}."
                )
            teacher_parameter.copy_(saved_parameter.to(dtype=teacher_parameter.dtype, device=teacher_parameter.device))

        saved_buffers = state_dict.get("buffers")
        if saved_buffers is None:
            self._copy_named_tensors(
                dict(self.actor_module.named_buffers()),
                dict(self.answer_teacher_module.named_buffers()),
            )
        else:
            self._copy_named_tensors(saved_buffers, dict(self.answer_teacher_module.named_buffers()))

    def _reshard_model_after_forward(self, model: nn.Module) -> None:
        if not torch.distributed.is_initialized():
            return
        process_group = self._get_fsdp_process_group(model)
        if torch.distributed.get_world_size(group=process_group) <= 1:
            return
        if isinstance(model, FSDP):
            model._handle.reshard(True)
        elif isinstance(model, FSDPModule):
            model.reshard()

    @contextmanager
    def _use_answer_teacher(self):
        """Temporarily run forwards with the EMA teacher, restoring the actor afterwards."""
        if self.answer_teacher_module is None:
            yield
            return

        actor_module = self.actor_module
        self.actor_module = self.answer_teacher_module
        self.actor_module.eval()
        try:
            yield
        finally:
            self._reshard_model_after_forward(self.actor_module)
            self.actor_module = actor_module

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        use_prefix_grouper: bool | None = None,
        use_remove_padding: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_remove_padding = self.use_remove_padding if use_remove_padding is None else use_remove_padding
        # PrefixGrouper path for shared-prefix optimization
        prefix_grouper_override = use_prefix_grouper is not None
        use_prefix_grouper = self.use_prefix_grouper if use_prefix_grouper is None else use_prefix_grouper
        if use_prefix_grouper:
            can_use_pg = (
                not use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and (prefix_grouper_override or not self.use_dynamic_bsz)
            )
            if can_use_pg and all(key in micro_batch for key in ("prompts", "response_mask", "uid")):
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                entropy, log_probs = forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )
                outputs = {"log_probs": log_probs}
                if calculate_entropy:
                    outputs["entropys"] = entropy
                return outputs

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    # logits = logits_rmpad.detach()
                    # logits, logits_indices = torch.topk(logits, k=100, dim=-1)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                    # logits = gather_outputs_and_unpad(
                    #     logits,
                    #     gather_dim=0,
                    #     unpad_dim=0,
                    #     padding_size=pad_size,
                    # )

                    # logits_indices = gather_outputs_and_unpad(
                    #     logits_indices,
                    #     gather_dim=0,
                    #     unpad_dim=0,
                    #     padding_size=pad_size,
                    # )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # full_logits = pad_input(
                #     hidden_states=logits,
                #     indices=indices,
                #     batch=batch_size,
                #     seqlen=seqlen,
                # )

                # full_logits_indices = pad_input(
                #     hidden_states=logits_indices,
                #     indices=indices,
                #     batch=batch_size,
                #     seqlen=seqlen,
                # )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                # logits = full_logits[:, -response_length - 1 : -1, :]
                # logits_indices = full_logits_indices[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            # outputs["logits"] = logits
            # outputs["logits_indices"] = logits_indices
            return outputs

    def _get_fsdp_process_group(self, model: nn.Module | None = None):
        model = self.actor_module if model is None else model
        process_group = getattr(model, "process_group", None)
        if process_group is None:
            process_group = getattr(model, "_process_group", None)
        if isinstance(process_group, tuple):
            process_group = process_group[0]
        return process_group

    @staticmethod
    def _get_micro_batch_sync_group():
        """Synchronize dynamic micro-batch counts across every actor rank.

        FSDP2 keeps its process groups in a DeviceMesh instead of exposing
        ``process_group`` on the module. For HSDP, synchronizing only the shard
        group is also insufficient because backward collectives span the
        replica dimension. Using WORLD keeps every rank's forward/backward
        collective sequence aligned for both FSDP and HSDP.
        """
        if not torch.distributed.is_initialized():
            return None
        return torch.distributed.group.WORLD

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor | float]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data, max_token_len=max_token_len, dp_group=self._get_micro_batch_sync_group()
            )
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        # logits_lst = []
        # logits_indices_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            # logits_lst.append(outputs["logits"])
            # logits_indices_lst.append(outputs["logits_indices"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        # logits = torch.concat(logits_lst, dim=0)
        # logits_indices = torch.concat(logits_indices_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)
        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        # outputs["logits"] = logits
        # outputs["logits_indices"] = logits_indices
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @staticmethod
    def _extract_ground_truth(reward_model_data) -> str:
        if isinstance(reward_model_data, dict):
            ground_truth = reward_model_data.get("ground_truth", "")
        else:
            ground_truth = getattr(reward_model_data, "ground_truth", "")
        return "" if ground_truth is None else str(ground_truth)

    @staticmethod
    def _decode_common_escapes(value: str) -> str:
        return str(value).replace("\\n", "\n").replace("\\t", "\t")

    @staticmethod
    def _as_float_scalar(value) -> float | None:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().reshape(-1)
            return None if value.numel() == 0 else float(value[0].item())
        if hasattr(value, "item"):
            value = value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_response_correct(data: DataProto, sample_idx: int) -> bool:
        if "acc" in data.non_tensor_batch:
            acc = DataParallelPPOActor._as_float_scalar(data.non_tensor_batch["acc"][sample_idx])
            if acc is not None:
                return acc > 0
        if "rm_scores" in data.batch.keys():
            return float(data.batch["rm_scores"][sample_idx].sum().item()) > 0
        raise ValueError("compute_answer_log_prob requires `acc` or `rm_scores` to select wrong answers.")

    @staticmethod
    def _strip_answer_prefix(text: str, answer_prefix: str) -> str:
        text = text.strip()
        prefix = answer_prefix.strip()
        if prefix and text[: len(prefix)].casefold() == prefix.casefold():
            return text[len(prefix) :].strip()
        return text

    @staticmethod
    def _last_boxed_content(text: str) -> str | None:
        left = "\\boxed{"
        start = text.rfind(left)
        if start < 0:
            return None
        content_start = start + len(left)
        depth = 1
        for idx in range(content_start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:idx].strip()
        return None

    @staticmethod
    def _extract_answer_from_response_text(response_text: str, answer_prefix: str) -> str:
        response_text = response_text.strip()
        if not response_text:
            return ""

        exact_prefix_idx = response_text.rfind(answer_prefix) if answer_prefix else -1
        if exact_prefix_idx >= 0:
            answer_tail = response_text[exact_prefix_idx + len(answer_prefix) :].strip()
            for line in answer_tail.splitlines():
                line = line.strip()
                if line:
                    return DataParallelPPOActor._strip_answer_prefix(line, answer_prefix)
            return ""

        prefix = answer_prefix.strip()
        if prefix:
            prefix_idx = response_text.casefold().rfind(prefix.casefold())
            if prefix_idx >= 0:
                answer_tail = response_text[prefix_idx + len(prefix) :].strip()
                for line in answer_tail.splitlines():
                    line = line.strip()
                    if line:
                        return DataParallelPPOActor._strip_answer_prefix(line, answer_prefix)
                return ""

        boxed = DataParallelPPOActor._last_boxed_content(response_text)
        if boxed:
            return boxed

        for line in reversed(response_text.splitlines()):
            line = line.strip()
            if line:
                return DataParallelPPOActor._strip_answer_prefix(line, answer_prefix)
        return response_text

    @staticmethod
    def _encode_prefixed_answer(tokenizer, answer_prefix: str, answer: str) -> tuple[list[int], list[int]]:
        answer = "" if answer is None else str(answer)
        full_text = answer_prefix + answer
        prefix_char_len = len(answer_prefix)

        try:
            encoded = tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            token_ids = list(encoded["input_ids"])
            offsets = encoded["offset_mapping"]
            answer_mask = [1 if end > prefix_char_len else 0 for _, end in offsets]
            if token_ids and (any(answer_mask) or not answer):
                return token_ids, answer_mask
        except (KeyError, NotImplementedError, TypeError, ValueError):
            pass

        prefix_ids = tokenizer.encode(answer_prefix, add_special_tokens=False)
        token_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prefix_token_len = min(len(prefix_ids), len(token_ids))
        answer_mask = [0] * prefix_token_len + [1] * (len(token_ids) - prefix_token_len)
        if token_ids and (any(answer_mask) or not answer):
            return token_ids, answer_mask

        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        return prefix_ids + answer_ids, [0] * len(prefix_ids) + [1] * len(answer_ids)

    def _get_delimiter_token_ids(self, tokenizer, delimiter: str) -> set[int]:
        cache_key = (id(tokenizer), delimiter)
        if cache_key in self._delimiter_token_id_cache:
            return self._delimiter_token_id_cache[cache_key]

        delimiter_token_ids: set[int] = set()
        if delimiter:
            vocab = tokenizer.get_vocab()
            for token_id in vocab.values():
                piece = tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if delimiter in piece:
                    delimiter_token_ids.add(token_id)
        self._delimiter_token_id_cache[cache_key] = delimiter_token_ids
        return delimiter_token_ids

    @staticmethod
    def _select_step_end_positions(
        tokenizer,
        valid_response_ids: list[int],
        valid_response_positions: list[int],
        delimiter_token_ids: set[int],
        delimiter_token_sequence: list[int],
        delimiter: str,
        step_interval: int,
        delimiter_step_marker_filter: bool,
        delimiter_step_marker_lookahead: int,
        delimiter_step_marker_patterns: list[str],
        delimiter_fallback_min_tokens: int,
        delimiter_max_steps_per_response: int,
    ) -> list[int]:
        delimiter_candidates = [
            (pos, idx, idx)
            for idx, (token_id, pos) in enumerate(zip(valid_response_ids, valid_response_positions, strict=True))
            if token_id in delimiter_token_ids
        ]
        if delimiter_token_sequence:
            seq_len = len(delimiter_token_sequence)
            for start in range(0, len(valid_response_ids) - seq_len + 1):
                if valid_response_ids[start : start + seq_len] == delimiter_token_sequence:
                    delimiter_candidates.append(
                        (valid_response_positions[start + seq_len - 1], start, start + seq_len - 1)
                    )

        candidates_by_position: dict[int, list[tuple[int, int]]] = {}
        for pos, start_idx, end_idx in delimiter_candidates:
            candidates_by_position.setdefault(pos, []).append((start_idx, end_idx))

        delimiter_positions = []
        boundary_candidates: list[tuple[int, int, bool]] = []
        for pos in sorted(candidates_by_position):
            if not delimiter_step_marker_filter:
                delimiter_positions.append(pos)
                continue

            candidate_ranges = candidates_by_position[pos]
            end_idx = max(end_idx for _, end_idx in candidate_ranges)
            is_strong_boundary = any(
                DataParallelPPOActor._has_strong_boundary_after_delimiter(
                    tokenizer=tokenizer,
                    valid_response_ids=valid_response_ids,
                    delimiter_start_idx=start_idx,
                    delimiter_end_idx=end_idx,
                    delimiter=delimiter,
                    lookahead=delimiter_step_marker_lookahead,
                    patterns=delimiter_step_marker_patterns,
                )
                for start_idx, end_idx in candidate_ranges
            )
            boundary_candidates.append((pos, end_idx, is_strong_boundary))

        if delimiter_step_marker_filter:
            strong_positions = [pos for pos, _, is_strong in boundary_candidates if is_strong]
            if strong_positions:
                delimiter_positions = strong_positions
            else:
                fallback_min_tokens = int(delimiter_fallback_min_tokens)
                if fallback_min_tokens > 0:
                    last_selected_end_idx = -1
                    for pos, end_idx, _ in boundary_candidates:
                        if end_idx - last_selected_end_idx >= fallback_min_tokens:
                            delimiter_positions.append(pos)
                            last_selected_end_idx = end_idx

        step_interval = max(int(step_interval), 1)
        selected_positions = []
        for step_idx, pos in enumerate(delimiter_positions, start=1):
            if step_idx % step_interval == 0 or step_idx == len(delimiter_positions):
                selected_positions.append(pos)

        max_steps = int(delimiter_max_steps_per_response)
        if max_steps > 0 and len(selected_positions) > max_steps:
            if max_steps == 1:
                return [selected_positions[-1]]
            last_idx = len(selected_positions) - 1
            selected_positions = [
                selected_positions[round(slot * last_idx / (max_steps - 1))] for slot in range(max_steps)
            ]
        return selected_positions

    @staticmethod
    def _has_strong_boundary_after_delimiter(
        tokenizer,
        valid_response_ids: list[int],
        delimiter_start_idx: int,
        delimiter_end_idx: int,
        delimiter: str,
        lookahead: int,
        patterns: list[str],
    ) -> bool:
        if lookahead <= 0 or not patterns:
            return False
        window_end = min(len(valid_response_ids), delimiter_end_idx + 1 + lookahead)
        window_ids = valid_response_ids[delimiter_start_idx:window_end]
        window_text = tokenizer.decode(
            window_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        delimiter_offset = window_text.find(delimiter) if delimiter else -1
        if delimiter_offset >= 0:
            marker_text = window_text[delimiter_offset + len(delimiter) :]
        else:
            marker_text = tokenizer.decode(
                valid_response_ids[delimiter_end_idx + 1 : window_end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        marker_text = marker_text.lstrip()
        return any(re.search(pattern, marker_text) is not None for pattern in patterns)

    @staticmethod
    def _pad_sequences(
        sequences: list[list[int]],
        pad_token_id: int,
        device: torch.device,
        dtype: torch.dtype,
        left_pad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(max((len(seq) for seq in sequences), default=0), 1)
        ids = torch.full((len(sequences), max_len), pad_token_id, dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            seq_tensor = torch.tensor(seq, dtype=dtype, device=device)
            if left_pad:
                ids[i, -len(seq) :] = seq_tensor
                mask[i, -len(seq) :] = 1
            else:
                ids[i, : len(seq)] = seq_tensor
                mask[i, : len(seq)] = 1
        return ids, mask

    def _build_answer_log_prob_data(
        self,
        prompt_prefixes: list[list[int]],
        answer_ids: list[list[int]],
        answer_masks: list[list[int]] | None,
        pad_token_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DataProto:
        prompt_ids, prompt_mask = self._pad_sequences(
            prompt_prefixes, pad_token_id=pad_token_id, device=device, dtype=dtype, left_pad=True
        )
        response_ids, response_mask = self._pad_sequences(
            answer_ids, pad_token_id=pad_token_id, device=device, dtype=dtype, left_pad=False
        )
        answer_loss_mask = None
        if answer_masks is not None:
            answer_loss_mask = torch.zeros_like(response_ids, dtype=torch.long)
            for i, answer_mask in enumerate(answer_masks):
                if len(answer_mask) != len(answer_ids[i]):
                    raise ValueError(
                        f"answer mask length {len(answer_mask)} does not match answer id length {len(answer_ids[i])}."
                    )
                if not answer_mask:
                    continue
                mask_tensor = torch.tensor(answer_mask, dtype=torch.long, device=device)
                answer_loss_mask[i, : len(answer_mask)] = mask_tensor
        input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
        position_ids = compute_position_id_with_mask(attention_mask)
        model_inputs = {
            "responses": response_ids,
            "response_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        if answer_loss_mask is not None:
            model_inputs["answer_loss_mask"] = answer_loss_mask
        return DataProto.from_dict(tensors=model_inputs)

    def _get_answer_dp_info(self):
        if not torch.distributed.is_initialized():
            return None, 0, 1
        process_group = self._get_fsdp_process_group()
        return (
            process_group,
            torch.distributed.get_rank(group=process_group),
            torch.distributed.get_world_size(group=process_group),
        )

    def _gather_answer_candidates(self, local_candidates: list[tuple[list[int], list[int], list[int], bool]]):
        process_group, rank, world_size = self._get_answer_dp_info()
        if world_size == 1:
            return local_candidates, rank, world_size

        device = torch.device(get_device_name(), get_device_id())
        metadata = []
        prompt_tokens = []
        answer_tokens = []
        answer_mask_tokens = []
        for prompt_ids, answer_ids, answer_mask, is_correct in local_candidates:
            metadata.append([len(prompt_ids), len(answer_ids), int(is_correct)])
            prompt_tokens.extend(prompt_ids)
            answer_tokens.extend(answer_ids)
            answer_mask_tokens.extend(answer_mask)

        metadata_tensor = (
            torch.tensor(metadata, dtype=torch.long, device=device).flatten()
            if metadata
            else torch.empty(0, dtype=torch.long, device=device)
        )
        prompt_tensor = (
            torch.tensor(prompt_tokens, dtype=torch.long, device=device)
            if prompt_tokens
            else torch.empty(0, dtype=torch.long, device=device)
        )
        answer_tensor = (
            torch.tensor(answer_tokens, dtype=torch.long, device=device)
            if answer_tokens
            else torch.empty(0, dtype=torch.long, device=device)
        )
        answer_mask_tensor = (
            torch.tensor(answer_mask_tokens, dtype=torch.long, device=device)
            if answer_mask_tokens
            else torch.empty(0, dtype=torch.long, device=device)
        )
        local_sizes = torch.tensor(
            [len(local_candidates), prompt_tensor.numel(), answer_tensor.numel(), answer_mask_tensor.numel()],
            dtype=torch.long,
            device=device,
        )
        gathered_sizes = [torch.empty_like(local_sizes) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_sizes, local_sizes, group=process_group)
        gathered_sizes = torch.stack(gathered_sizes)
        max_item_count = int(gathered_sizes[:, 0].max().item())
        max_prompt_tokens = int(gathered_sizes[:, 1].max().item())
        max_answer_tokens = int(gathered_sizes[:, 2].max().item())
        max_answer_mask_tokens = int(gathered_sizes[:, 3].max().item())
        local_payload = torch.cat([metadata_tensor, prompt_tensor, answer_tensor, answer_mask_tensor], dim=0)
        max_payload_len = max_item_count * 3 + max_prompt_tokens + max_answer_tokens + max_answer_mask_tokens

        def _pad_1d(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
            if tensor.numel() == target_len:
                return tensor
            padded = torch.zeros(target_len, dtype=tensor.dtype, device=tensor.device)
            if tensor.numel() > 0:
                padded[: tensor.numel()] = tensor
            return padded

        local_payload = _pad_1d(local_payload, max_payload_len)
        gathered_payloads = [torch.empty_like(local_payload) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_payloads, local_payload, group=process_group)

        global_candidates = []
        gathered_sizes = gathered_sizes.cpu()
        for rank_idx in range(world_size):
            item_count = int(gathered_sizes[rank_idx, 0].item())
            prompt_token_count = int(gathered_sizes[rank_idx, 1].item())
            answer_token_count = int(gathered_sizes[rank_idx, 2].item())
            answer_mask_token_count = int(gathered_sizes[rank_idx, 3].item())
            metadata_len = item_count * 3
            prompt_start = metadata_len
            answer_start = prompt_start + prompt_token_count
            answer_mask_start = answer_start + answer_token_count
            rank_payload = gathered_payloads[rank_idx]
            rank_metadata = rank_payload[:metadata_len].view(item_count, 3).cpu().tolist()
            rank_prompts = rank_payload[prompt_start:answer_start].cpu().tolist()
            rank_answers = rank_payload[answer_start : answer_start + answer_token_count].cpu().tolist()
            rank_answer_masks = rank_payload[
                answer_mask_start : answer_mask_start + answer_mask_token_count
            ].cpu().tolist()

            prompt_offset = 0
            answer_offset = 0
            answer_mask_offset = 0
            for prompt_len, answer_len, is_correct in rank_metadata:
                prompt_ids = rank_prompts[prompt_offset : prompt_offset + prompt_len]
                answer_ids = rank_answers[answer_offset : answer_offset + answer_len]
                answer_mask = rank_answer_masks[answer_mask_offset : answer_mask_offset + answer_len]
                prompt_offset += prompt_len
                answer_offset += answer_len
                answer_mask_offset += answer_len
                global_candidates.append((prompt_ids, answer_ids, answer_mask, bool(is_correct)))
        return global_candidates, rank, world_size

    def _gather_answer_items(
        self, local_items: list[tuple[list[int], list[int], list[int], tuple[int, int, int, int]]]
    ):
        process_group, rank, world_size = self._get_answer_dp_info()
        if world_size == 1:
            return local_items, rank, world_size

        device = torch.device(get_device_name(), get_device_id())
        metadata = []
        prompt_tokens = []
        answer_tokens = []
        answer_mask_tokens = []
        for prompt_ids, answer_ids, answer_mask, (owner_rank, sample_idx, answer_pos, answer_sign) in local_items:
            metadata.append([len(prompt_ids), len(answer_ids), owner_rank, sample_idx, answer_pos, answer_sign])
            prompt_tokens.extend(prompt_ids)
            answer_tokens.extend(answer_ids)
            answer_mask_tokens.extend(answer_mask)

        metadata_tensor = (
            torch.tensor(metadata, dtype=torch.long, device=device).flatten()
            if metadata
            else torch.empty(0, dtype=torch.long, device=device)
        )
        prompt_tensor = (
            torch.tensor(prompt_tokens, dtype=torch.long, device=device)
            if prompt_tokens
            else torch.empty(0, dtype=torch.long, device=device)
        )
        answer_tensor = (
            torch.tensor(answer_tokens, dtype=torch.long, device=device)
            if answer_tokens
            else torch.empty(0, dtype=torch.long, device=device)
        )
        answer_mask_tensor = (
            torch.tensor(answer_mask_tokens, dtype=torch.long, device=device)
            if answer_mask_tokens
            else torch.empty(0, dtype=torch.long, device=device)
        )
        local_sizes = torch.tensor(
            [len(local_items), prompt_tensor.numel(), answer_tensor.numel(), answer_mask_tensor.numel()],
            dtype=torch.long,
            device=device,
        )
        gathered_sizes = [torch.empty_like(local_sizes) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_sizes, local_sizes, group=process_group)
        gathered_sizes = torch.stack(gathered_sizes)
        max_item_count = int(gathered_sizes[:, 0].max().item())
        max_prompt_tokens = int(gathered_sizes[:, 1].max().item())
        max_answer_tokens = int(gathered_sizes[:, 2].max().item())
        max_answer_mask_tokens = int(gathered_sizes[:, 3].max().item())
        local_payload = torch.cat([metadata_tensor, prompt_tensor, answer_tensor, answer_mask_tensor], dim=0)
        max_payload_len = max_item_count * 6 + max_prompt_tokens + max_answer_tokens + max_answer_mask_tokens

        def _pad_1d(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
            if tensor.numel() == target_len:
                return tensor
            padded = torch.zeros(target_len, dtype=tensor.dtype, device=tensor.device)
            if tensor.numel() > 0:
                padded[: tensor.numel()] = tensor
            return padded

        local_payload = _pad_1d(local_payload, max_payload_len)
        gathered_payloads = [torch.empty_like(local_payload) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_payloads, local_payload, group=process_group)

        global_items = []
        gathered_sizes = gathered_sizes.cpu()
        for rank_idx in range(world_size):
            item_count = int(gathered_sizes[rank_idx, 0].item())
            prompt_token_count = int(gathered_sizes[rank_idx, 1].item())
            answer_token_count = int(gathered_sizes[rank_idx, 2].item())
            answer_mask_token_count = int(gathered_sizes[rank_idx, 3].item())
            metadata_len = item_count * 6
            prompt_start = metadata_len
            answer_start = prompt_start + prompt_token_count
            answer_mask_start = answer_start + answer_token_count
            rank_payload = gathered_payloads[rank_idx]
            rank_metadata = rank_payload[:metadata_len].view(item_count, 6).cpu().tolist()
            rank_prompts = rank_payload[prompt_start:answer_start].cpu().tolist()
            rank_answers = rank_payload[answer_start : answer_start + answer_token_count].cpu().tolist()
            rank_answer_masks = rank_payload[
                answer_mask_start : answer_mask_start + answer_mask_token_count
            ].cpu().tolist()

            prompt_offset = 0
            answer_offset = 0
            answer_mask_offset = 0
            for prompt_len, answer_len, owner_rank, sample_idx, answer_pos, answer_sign in rank_metadata:
                prompt_ids = rank_prompts[prompt_offset : prompt_offset + prompt_len]
                answer_ids = rank_answers[answer_offset : answer_offset + answer_len]
                answer_mask = rank_answer_masks[answer_mask_offset : answer_mask_offset + answer_len]
                prompt_offset += prompt_len
                answer_offset += answer_len
                answer_mask_offset += answer_len
                global_items.append(
                    (prompt_ids, answer_ids, answer_mask, (owner_rank, sample_idx, answer_pos, answer_sign))
                )
        return global_items, rank, world_size

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_answer_log_prob(self, data: DataProto, tokenizer) -> dict[str, torch.Tensor]:
        """Compute correct-answer log probabilities or correct-minus-wrong margins.

        For every response prefix, the correct answer score is its mean answer-token
        log probability. When ``answer_log_prob_num_wrong_answers`` is positive,
        subtract the mean score of up to that many distinct wrong answers sampled
        from responses belonging to the same prompt.

        Returns:
            dict[str, torch.Tensor]:
                log_probs: (batch_size, response_length + 1)
                step_mask: (batch_size, response_length)
        """
        self.actor_module.eval()

        # if "multi_modal_inputs" in data.non_tensor_batch.keys():
        #     raise NotImplementedError("compute_answer_log_prob currently supports text-only batches.")

        micro_batch_size = max(int(data.meta_info["micro_batch_size"]), 1)
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        num_wrong_answers = int(
            data.meta_info.get(
                "answer_log_prob_num_wrong_answers",
                self.config.get("answer_log_prob_num_wrong_answers", 1),
            )
        )
        if num_wrong_answers < 0:
            raise ValueError("answer_log_prob_num_wrong_answers must be non-negative.")
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        delimiter = self._decode_common_escapes(data.meta_info["delimiter"])
        answer_prefix = self._decode_common_escapes(data.meta_info["answer_prefix"])
        step_interval = data.meta_info["step_interval"]
        delimiter_step_marker_filter = data.meta_info.get("delimiter_step_marker_filter", False)
        delimiter_step_marker_lookahead = int(data.meta_info.get("delimiter_step_marker_lookahead", 10))
        delimiter_fallback_min_tokens = int(
            data.meta_info.get(
                "delimiter_fallback_min_tokens",
                self.config.get("delimiter_fallback_min_tokens", 0),
            )
        )
        delimiter_max_steps_per_response = int(
            data.meta_info.get(
                "delimiter_max_steps_per_response",
                self.config.get("delimiter_max_steps_per_response", 0),
            )
        )
        delimiter_step_marker_patterns = [
            self._decode_common_escapes(pattern)
            for pattern in data.meta_info.get(
                "delimiter_step_marker_patterns",
                [
                    r"(?i)\bStep\s*\d+\b",
                    r"\b\d+\.\s",
                    (
                        r"(?i)^(?:[#*_]+[ \t]+)*[#*_]*[ \t]*"
                        r"(?:First|Firstly|Second|Secondly|Third|Thirdly|Next|Then|Finally|Similarly)\b"
                    ),
                ],
            )
        ]

        select_keys = ["responses", "input_ids", "attention_mask", "response_mask"]
        if num_wrong_answers > 0 and "rm_scores" in data.batch.keys():
            select_keys.append("rm_scores")
        non_tensor_select_keys = ["reward_model"]
        if num_wrong_answers > 0 and "acc" in data.non_tensor_batch:
            non_tensor_select_keys.append("acc")
        if num_wrong_answers > 0 and "pred" in data.non_tensor_batch:
            non_tensor_select_keys.append("pred")
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        responses = data.batch["responses"]
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        response_mask = data.batch["response_mask"]
        response_len = responses.size(1)
        prompt_len = input_ids.size(1) - response_len
        batch_size = responses.size(0)
        device = torch.device(get_device_name(), get_device_id())
        dtype = responses.dtype

        delimiter_token_ids = self._get_delimiter_token_ids(tokenizer, delimiter)
        delimiter_token_sequence = tokenizer.encode(delimiter, add_special_tokens=False) if delimiter else []
        answer_log_probs = torch.zeros((batch_size, response_len + 1), dtype=torch.float32, device=device)
        step_mask = torch.zeros((batch_size, response_len), dtype=torch.long, device=device)
        computed_pos_mask = torch.zeros((batch_size, response_len + 1), dtype=torch.bool, device=device)
        computed_pos_mask[:, 0] = True

        process_group, rank, world_size = self._get_answer_dp_info()
        sample_records = []
        local_candidates: list[tuple[list[int], list[int], list[int], bool]] = []
        ground_truth_encoding_cache: dict[str, tuple[list[int], list[int]]] = {}

        for i in range(batch_size):
            prompt_token_ids = input_ids[i, :prompt_len][attention_mask[i, :prompt_len].bool()].tolist()
            valid_response_positions = response_mask[i].nonzero(as_tuple=False).flatten().tolist()
            valid_response_ids = responses[i, valid_response_positions].tolist()
            selected_positions = self._select_step_end_positions(
                tokenizer=tokenizer,
                valid_response_ids=valid_response_ids,
                valid_response_positions=valid_response_positions,
                delimiter_token_ids=delimiter_token_ids,
                delimiter_token_sequence=delimiter_token_sequence,
                delimiter=delimiter,
                step_interval=step_interval,
                delimiter_step_marker_filter=delimiter_step_marker_filter,
                delimiter_step_marker_lookahead=delimiter_step_marker_lookahead,
                delimiter_step_marker_patterns=delimiter_step_marker_patterns,
                delimiter_fallback_min_tokens=delimiter_fallback_min_tokens,
                delimiter_max_steps_per_response=delimiter_max_steps_per_response,
            )
            selected_prefix_lengths = [
                bisect_right(valid_response_positions, position) for position in selected_positions
            ]

            reward_model_data = data.non_tensor_batch["reward_model"][i]
            ground_truth = self._extract_ground_truth(reward_model_data)
            if ground_truth not in ground_truth_encoding_cache:
                ground_truth_encoding_cache[ground_truth] = self._encode_prefixed_answer(
                    tokenizer,
                    answer_prefix,
                    ground_truth,
                )
            encoded_answer, encoded_answer_mask = ground_truth_encoding_cache[ground_truth]
            encoded_wrong_answer = None
            encoded_wrong_answer_mask = None
            is_correct = None
            if num_wrong_answers > 0:
                wrong_answer = ""
                if "pred" in data.non_tensor_batch:
                    pred = data.non_tensor_batch["pred"][i]
                    if hasattr(pred, "item"):
                        pred = pred.item()
                    if pred is not None:
                        pred = str(pred).strip()
                        if pred and pred != "[INVALID]":
                            wrong_answer = self._strip_answer_prefix(pred, answer_prefix)
                if not wrong_answer:
                    response_text = tokenizer.decode(
                        valid_response_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    wrong_answer = self._extract_answer_from_response_text(response_text, answer_prefix)
                encoded_wrong_answer, encoded_wrong_answer_mask = self._encode_prefixed_answer(
                    tokenizer,
                    answer_prefix,
                    wrong_answer,
                )
                is_correct = self._is_response_correct(data, i)

            sample_records.append(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "valid_response_ids": valid_response_ids,
                    "selected_positions": selected_positions,
                    "selected_prefix_lengths": selected_prefix_lengths,
                    "encoded_answer": encoded_answer,
                    "encoded_answer_mask": encoded_answer_mask,
                    "encoded_wrong_answer": encoded_wrong_answer,
                    "encoded_wrong_answer_mask": encoded_wrong_answer_mask,
                    "is_correct": is_correct,
                }
            )
            if num_wrong_answers > 0:
                local_candidates.append((prompt_token_ids, encoded_wrong_answer, encoded_wrong_answer_mask, is_correct))

        wrong_answers_by_prompt: dict[tuple[int, ...], list[tuple[list[int], list[int]]]] = {}
        selected_wrong_answers_by_prompt: dict[tuple[int, ...], list[tuple[list[int], list[int]]]] = {}
        if num_wrong_answers > 0:
            global_candidates, _, _ = self._gather_answer_candidates(local_candidates)
            seen_wrong_answers_by_prompt: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
            for prompt_token_ids, wrong_answer_ids, wrong_answer_mask, is_correct in global_candidates:
                if is_correct:
                    continue
                prompt_key = tuple(prompt_token_ids)
                answer_key = tuple(wrong_answer_ids)
                seen_answers = seen_wrong_answers_by_prompt.setdefault(prompt_key, set())
                if answer_key in seen_answers:
                    continue
                seen_answers.add(answer_key)
                wrong_answers_by_prompt.setdefault(prompt_key, []).append((wrong_answer_ids, wrong_answer_mask))

            if world_size > 1:
                sample_seed = torch.zeros(1, dtype=torch.long, device=device)
                if rank == 0:
                    sample_seed[0] = int(np.random.randint(0, np.iinfo(np.int32).max))
                torch.distributed.all_reduce(
                    sample_seed,
                    op=torch.distributed.ReduceOp.MAX,
                    group=process_group,
                )
                sampling_rng = np.random.default_rng(int(sample_seed.item()))
            else:
                sampling_rng = np.random
            for prompt_key, wrong_answers in wrong_answers_by_prompt.items():
                selected_count = min(num_wrong_answers, len(wrong_answers))
                if selected_count:
                    selected_indices = sampling_rng.choice(len(wrong_answers), size=selected_count, replace=False)
                    selected_wrong_answers_by_prompt[prompt_key] = [
                        wrong_answers[idx] for idx in selected_indices.tolist()
                    ]

        local_items: list[tuple[list[int], list[int], list[int], tuple[int, int, int, int]]] = []
        step_sample_indices: list[int] = []
        step_positions: list[int] = []
        for i, sample_record in enumerate(sample_records):
            prompt_token_ids = sample_record["prompt_token_ids"]
            valid_response_ids = sample_record["valid_response_ids"]
            selected_positions = sample_record["selected_positions"]
            selected_prefix_lengths = sample_record["selected_prefix_lengths"]
            encoded_answer = sample_record["encoded_answer"]
            encoded_answer_mask = sample_record["encoded_answer_mask"]
            local_items.append((prompt_token_ids, encoded_answer, encoded_answer_mask, (rank, i, 0, 1)))
            selected_wrong_answers = selected_wrong_answers_by_prompt.get(tuple(prompt_token_ids), [])
            for wrong_answer_ids, wrong_answer_mask in selected_wrong_answers:
                local_items.append((prompt_token_ids, wrong_answer_ids, wrong_answer_mask, (rank, i, 0, -1)))

            step_sample_indices.extend([i] * len(selected_positions))
            step_positions.extend(selected_positions)
            for pos, prefix_len in zip(selected_positions, selected_prefix_lengths, strict=True):
                prefix_token_ids = prompt_token_ids + valid_response_ids[:prefix_len]
                local_items.append(
                    (prefix_token_ids, encoded_answer, encoded_answer_mask, (rank, i, pos + 1, 1))
                )
                for wrong_answer_ids, wrong_answer_mask in selected_wrong_answers:
                    local_items.append(
                        (prefix_token_ids, wrong_answer_ids, wrong_answer_mask, (rank, i, pos + 1, -1))
                    )

        if step_positions:
            step_indices = torch.tensor(
                [step_sample_indices, step_positions],
                dtype=torch.long,
                device=device,
            )
            step_mask[step_indices[0], step_indices[1]] = 1
            computed_pos_mask[step_indices[0], step_indices[1] + 1] = True

        global_items, rank, world_size = self._gather_answer_items(local_items)
        assert len(global_items) > 0, "compute_answer_log_prob expects at least one derived answer item."

        prompt_prefixes = [item[0] for item in global_items]
        answer_ids = [item[1] for item in global_items]
        answer_masks = [item[2] for item in global_items]
        answer_data = self._build_answer_log_prob_data(
            prompt_prefixes=prompt_prefixes,
            answer_ids=answer_ids,
            answer_masks=answer_masks,
            pad_token_id=pad_token_id,
            device=device,
            dtype=dtype,
        )
        answer_data.meta_info.update(data.meta_info)
        answer_data.meta_info["use_dynamic_bsz"] = use_dynamic_bsz

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            max_token_len = max(max_token_len, answer_data.batch["attention_mask"].shape[-1])
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                answer_data,
                max_token_len=max_token_len,
                dp_group=None,
                same_micro_num_in_dp=False,
                use_karmarkar_karp=False,
            )
        else:
            micro_batches = answer_data.split(micro_batch_size)
            batch_idx_list = [
                list(range(start, min(start + micro_batch_size, len(answer_data))))
                for start in range(0, len(answer_data), micro_batch_size)
            ]

        global_micro_batch_count = len(micro_batches)
        micro_batch_count_per_rank = (global_micro_batch_count + world_size - 1) // world_size
        padded_micro_batch_count = micro_batch_count_per_rank * world_size
        dummy_micro_batch = answer_data[:1]
        replicated_compute = self.use_ulysses_sp

        global_answer_log_probs = torch.zeros(len(global_items), dtype=torch.float32, device=device)
        if replicated_compute:
            micro_batch_indices = range(global_micro_batch_count)
        else:
            micro_batch_indices = range(rank, padded_micro_batch_count, world_size)

        with self._use_answer_teacher():
            for micro_batch_idx in micro_batch_indices:
                is_dummy_micro_batch = (not replicated_compute) and micro_batch_idx >= global_micro_batch_count
                micro_batch = dummy_micro_batch if is_dummy_micro_batch else micro_batches[micro_batch_idx]
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                with torch.no_grad():
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                    )
                if is_dummy_micro_batch:
                    continue
                answer_mask = model_inputs.get("answer_loss_mask", model_inputs["response_mask"]).to(
                    outputs["log_probs"].dtype
                )
                answer_token_count = answer_mask.sum(dim=-1).clamp_min(1.0)
                answer_log_prob = (outputs["log_probs"] * answer_mask).sum(dim=-1) / answer_token_count
                global_item_indices = torch.tensor(batch_idx_list[micro_batch_idx], dtype=torch.long, device=device)
                global_answer_log_probs[global_item_indices] = answer_log_prob.to(torch.float32)

        if world_size > 1 and not self.use_ulysses_sp:
            torch.distributed.all_reduce(
                global_answer_log_probs,
                op=torch.distributed.ReduceOp.SUM,
                group=self._get_fsdp_process_group(),
            )
        owned_item_indices = []
        owned_sample_indices = []
        owned_answer_positions = []
        owned_wrong_item_indices: dict[tuple[int, int], list[int]] = {}
        for global_item_idx, (_, _, _, (owner_rank, sample_idx, answer_pos, answer_sign)) in enumerate(global_items):
            if owner_rank == rank:
                if answer_sign > 0:
                    owned_item_indices.append(global_item_idx)
                    owned_sample_indices.append(sample_idx)
                    owned_answer_positions.append(answer_pos)
                else:
                    owned_wrong_item_indices.setdefault((sample_idx, answer_pos), []).append(global_item_idx)

        if owned_item_indices:
            owned_item_indices = torch.tensor(owned_item_indices, dtype=torch.long, device=device)
            owned_sample_indices = torch.tensor(owned_sample_indices, dtype=torch.long, device=device)
            owned_answer_positions = torch.tensor(owned_answer_positions, dtype=torch.long, device=device)
            owned_answer_log_probs = global_answer_log_probs[owned_item_indices]
            answer_log_probs.index_put_(
                (owned_sample_indices, owned_answer_positions),
                owned_answer_log_probs,
                accumulate=True,
            )
        for (sample_idx, answer_pos), wrong_item_indices in owned_wrong_item_indices.items():
            wrong_item_indices = torch.tensor(wrong_item_indices, dtype=torch.long, device=device)
            answer_log_probs[sample_idx, answer_pos] -= global_answer_log_probs[wrong_item_indices].mean()

        answer_pos_ids = torch.arange(response_len + 1, dtype=torch.long, device=device).unsqueeze(0)
        last_computed_pos = torch.where(computed_pos_mask, answer_pos_ids, torch.zeros_like(answer_pos_ids))
        last_computed_pos = last_computed_pos.cummax(dim=1).values
        answer_log_probs = answer_log_probs.gather(dim=1, index=last_computed_pos)

        return {"log_probs": answer_log_probs, "step_mask": step_mask}

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        kl_loss_needs_ref = self.config.use_kl_loss and float(self.config.kl_loss_coef) != 0.0
        if kl_loss_needs_ref or loss_mode in {"ours", "my", "my_future"}:
            select_keys.append("ref_log_prob")
            # select_keys.append("ref_logits")
            # select_keys.append("ref_logits_indices")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "vinfo_weights" in data.batch.keys():
            select_keys.append("vinfo_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len, dp_group=self._get_micro_batch_sync_group()
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = (
                        self.config.calculate_entropy
                        or (entropy_coeff != 0)
                        or (self.config.policy_loss.get("loss_mode", "vanilla") == "ours")
                    )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    if loss_mode == "dgpo":
                        ref_logits = model_inputs["ref_logits"]
                        ref_logits_indices = model_inputs["ref_logits_indices"]
                        logits = outputs["logits"]
                        logits_indices = outputs["logits_indices"]
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            logits=logits,
                            ref_logits=ref_logits,
                            logits_indices=logits_indices,
                            ref_logits_indices=ref_logits_indices,
                            entropy=entropy,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                    elif loss_mode == "ours":
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            w=model_inputs["vinfo_weights"],
                            ref_log_prob=model_inputs["ref_log_prob"],
                            entropy=entropy,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                    elif loss_mode == "my" or loss_mode == "my_future":
                        ref_log_prob = model_inputs["ref_log_prob"]
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            ref_log_prob=ref_log_prob,
                            entropy=entropy,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                    else:
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if kl_loss_needs_ref:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        self.update_answer_teacher()
        return metrics
