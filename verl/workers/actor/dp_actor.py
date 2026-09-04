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
import math
import os
from contextlib import contextmanager, nullcontext

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
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = [
    "DataParallelPPOActor",
]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CounterfactualCreditHead(nn.Module):
    """Predict a bounded state value from a boundary hidden state."""

    def __init__(self, hidden_size: int, projection_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.projection_size = int(projection_size)
        self.input_layer = nn.Linear(self.hidden_size, self.projection_size, dtype=torch.float32)
        self.activation = nn.SiLU()
        self.output_layer = nn.Linear(self.projection_size, 1, dtype=torch.float32)
        self._reset_output_parameters()

    def _reset_output_parameters(self) -> None:
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_layer.bias)

    def reset_parameters(self) -> None:
        self.input_layer.reset_parameters()
        self._reset_output_parameters()

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        logits = self.output_layer(self.activation(self.input_layer(representation)))
        return torch.sigmoid(logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
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

        self._init_counterfactual_credit_head()

    @staticmethod
    def _unwrap_model(module: nn.Module) -> nn.Module:
        """Unwrap only transparent distributed/PEFT containers."""
        seen = set()
        while id(module) not in seen:
            seen.add(id(module))
            if isinstance(module, FSDP):
                module = module._fsdp_wrapped_module
                continue
            wrapped_module = getattr(module, "module", None)
            if isinstance(wrapped_module, nn.Module) and wrapped_module is not module:
                module = wrapped_module
                continue
            break
        return module

    @staticmethod
    def _model_hidden_size(module: nn.Module) -> int:
        module = DataParallelPPOActor._unwrap_model(module)
        config = getattr(module, "config", None)
        candidate_configs = [config, getattr(config, "text_config", None)]
        for candidate in candidate_configs:
            if candidate is None:
                continue
            for field_name in ("hidden_size", "n_embd", "d_model"):
                value = getattr(candidate, field_name, None)
                if value is not None:
                    return int(value)
        raise ValueError("Could not infer the actor hidden size from the model config.")

    @staticmethod
    def _model_vocab_size(module: nn.Module) -> int:
        module = DataParallelPPOActor._unwrap_model(module)
        config = getattr(module, "config", None)
        candidate_configs = [
            config,
            getattr(config, "text_config", None),
            getattr(config, "language_config", None),
        ]
        for candidate in candidate_configs:
            vocab_size = getattr(candidate, "vocab_size", None)
            if vocab_size is not None and int(vocab_size) > 1:
                return int(vocab_size)

        vocab_size = getattr(module, "vocab_size", None)
        if vocab_size is not None and int(vocab_size) > 1:
            return int(vocab_size)

        output_embeddings = getattr(module, "get_output_embeddings", lambda: None)()
        weight = getattr(output_embeddings, "weight", None)
        if weight is not None and weight.ndim >= 1 and int(weight.shape[0]) > 1:
            return int(weight.shape[0])

        raise ValueError("Could not infer the actor vocabulary size for DGPO entropy normalization.")

    def _init_counterfactual_credit_head(self) -> None:
        head_config = self.config.get("counterfactual_credit_head", None)
        self.counterfactual_credit_head_enabled = bool(head_config is not None and head_config.get("enabled", False))
        self.counterfactual_credit_head = None
        self.counterfactual_credit_optimizer = None
        self.counterfactual_credit_updates = 0
        self.counterfactual_credit_last_global_step = None
        self._counterfactual_credit_needs_broadcast = False
        if not self.counterfactual_credit_head_enabled:
            return
        if self.actor_optimizer is None:
            raise ValueError("counterfactual_credit_head is supported only by the trainable actor.")
        if self.use_ulysses_sp:
            raise ValueError("counterfactual_credit_head does not support Ulysses sequence parallelism (SP > 1).")
        if self.use_prefix_grouper:
            raise ValueError("counterfactual_credit_head does not support use_prefix_grouper=True.")
        if self.use_fused_kernels:
            raise ValueError("counterfactual_credit_head does not support use_fused_kernels=True.")

        model = self._unwrap_model(self.actor_module)
        if getattr(getattr(model, "config", None), "vision_config", None) is not None:
            raise ValueError("counterfactual_credit_head currently supports text-only causal language models.")
        if not isinstance(getattr(model, "get_output_embeddings", lambda: None)(), nn.Module):
            raise ValueError("counterfactual_credit_head requires an accessible output-embedding module.")
        hidden_size = self._model_hidden_size(model)
        with torch.random.fork_rng(devices=[], device_type="cpu"):
            self.counterfactual_credit_head = CounterfactualCreditHead(
                hidden_size,
                int(head_config.get("hidden_dim", 512)),
            )
        self._counterfactual_credit_needs_broadcast = True
        self.counterfactual_credit_optimizer = torch.optim.AdamW(
            self.counterfactual_credit_head.parameters(),
            lr=float(head_config.get("lr", 1e-3)),
            weight_decay=float(head_config.get("weight_decay", 0.0)),
        )

    @property
    def has_counterfactual_credit_head(self) -> bool:
        return self.counterfactual_credit_head is not None

    @staticmethod
    def _move_optimizer_value(value, device: torch.device):
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, dict):
            return {key: DataParallelPPOActor._move_optimizer_value(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [DataParallelPPOActor._move_optimizer_value(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(DataParallelPPOActor._move_optimizer_value(item, device) for item in value)
        return value

    def _ensure_counterfactual_credit_device(self, device: torch.device) -> None:
        if not self.has_counterfactual_credit_head:
            return
        self.counterfactual_credit_head.to(device=device, dtype=torch.float32)
        for state in self.counterfactual_credit_optimizer.state.values():
            for key, value in list(state.items()):
                state[key] = self._move_optimizer_value(value, device)
        if self._counterfactual_credit_needs_broadcast and torch.distributed.is_initialized():
            for parameter in self.counterfactual_credit_head.parameters():
                torch.distributed.broadcast(parameter.data, src=0)
            self._counterfactual_credit_needs_broadcast = False

    def _get_output_embeddings_module(self) -> nn.Module:
        model = self._unwrap_model(self.actor_module)
        output_embeddings = getattr(model, "get_output_embeddings", lambda: None)()
        if not isinstance(output_embeddings, nn.Module):
            raise RuntimeError("Final-hidden capture lost access to the actor output-embedding module.")
        return output_embeddings

    @contextmanager
    def _capture_final_hidden_state(self, selector: torch.Tensor):
        """Capture the lm-head input without asking Transformers for every layer."""
        captured = []

        def capture_input(_module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Final-hidden capture expected the lm-head input to be a tensor.")
            hidden = args[0]
            if selector.shape != hidden.shape[:-1]:
                raise RuntimeError(
                    "Final-hidden selector does not match the lm-head input: "
                    f"selector={tuple(selector.shape)}, hidden={tuple(hidden.shape)}."
                )
            captured.append(hidden[selector].detach())

        handle = self._get_output_embeddings_module().register_forward_pre_hook(capture_input)
        try:
            yield captured
        finally:
            handle.remove()

    @staticmethod
    def _take_captured_hidden(captured: list[torch.Tensor]) -> torch.Tensor:
        if len(captured) != 1:
            raise RuntimeError(
                "Final-hidden capture expected exactly one lm-head invocation per actor forward, "
                f"but observed {len(captured)}."
            )
        hidden = captured[0]
        if not hidden.is_floating_point() or hidden.ndim != 2:
            raise RuntimeError("Expected selected final hidden states with shape [steps, hidden].")
        return hidden

    @staticmethod
    def _nested_to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: DataParallelPPOActor._nested_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [DataParallelPPOActor._nested_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(DataParallelPPOActor._nested_to_cpu(item) for item in value)
        return value

    def counterfactual_credit_state_dict(self) -> dict | None:
        if not self.has_counterfactual_credit_head:
            return None
        return {
            "format_version": 2,
            "architecture": {
                "type": "boundary_value_two_layer_mlp_sigmoid",
                "hidden_size": self.counterfactual_credit_head.hidden_size,
                "projection_size": self.counterfactual_credit_head.projection_size,
            },
            "model": {
                name: tensor.detach().cpu() for name, tensor in self.counterfactual_credit_head.state_dict().items()
            },
            "optimizer": self._nested_to_cpu(self.counterfactual_credit_optimizer.state_dict()),
            "updates": self.counterfactual_credit_updates,
            "last_global_step": self.counterfactual_credit_last_global_step,
        }

    def load_counterfactual_credit_state_dict(self, state_dict: dict | None) -> None:
        if not self.has_counterfactual_credit_head:
            return
        if state_dict is None:
            head_device = next(self.counterfactual_credit_head.parameters()).device
            fork_devices = [] if head_device.type == "cpu" else [head_device]
            with torch.random.fork_rng(devices=fork_devices, device_type=head_device.type):
                self.counterfactual_credit_head.reset_parameters()
            self.counterfactual_credit_optimizer.state.clear()
            self.counterfactual_credit_updates = 0
            self.counterfactual_credit_last_global_step = None
            self._counterfactual_credit_needs_broadcast = True
            return
        expected = {
            "type": "boundary_value_two_layer_mlp_sigmoid",
            "hidden_size": self.counterfactual_credit_head.hidden_size,
            "projection_size": self.counterfactual_credit_head.projection_size,
        }
        format_version = int(state_dict.get("format_version", 0))
        architecture = state_dict.get("architecture")
        is_legacy_direct_credit = (
            format_version == 1
            and isinstance(architecture, dict)
            and architecture.get("type") == "h_pre_delta_two_layer_mlp"
        )
        if is_legacy_direct_credit:
            logger.warning(
                "Resetting the legacy direct-credit probe while retaining the resumed actor checkpoint; "
                "its 2H transition input is incompatible with the new H-dimensional value probe."
            )
            self.load_counterfactual_credit_state_dict(None)
            return
        if format_version != 2 or architecture != expected:
            raise RuntimeError("Counterfactual-credit checkpoint architecture does not match the configured head.")
        self.counterfactual_credit_head.load_state_dict(state_dict["model"], strict=True)
        self.counterfactual_credit_optimizer.load_state_dict(state_dict["optimizer"])
        self.counterfactual_credit_updates = int(state_dict.get("updates", 0))
        self.counterfactual_credit_last_global_step = state_dict.get("last_global_step")
        self._counterfactual_credit_needs_broadcast = False

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

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_credit_boundary_hidden: bool = False,
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
                if return_credit_boundary_hidden is True:
                    credit_boundary_hidden: the last prompt hidden followed by
                        every semantic step-end hidden for each response.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_remove_padding = self.use_remove_padding if use_remove_padding is None else use_remove_padding
        capture_hidden = return_credit_boundary_hidden
        if capture_hidden:
            if self.use_ulysses_sp:
                raise ValueError("Credit-boundary capture does not support Ulysses sequence parallelism (SP > 1).")
            if self.use_fused_kernels:
                raise ValueError("Credit-boundary capture does not support use_fused_kernels=True.")
            if self.use_prefix_grouper:
                raise ValueError("Credit-boundary capture does not support use_prefix_grouper=True.")
            if "step_end_mask" not in micro_batch:
                raise ValueError("Credit-boundary capture requires step_end_mask in the actor batch.")
        # PrefixGrouper path for shared-prefix optimization
        prefix_grouper_override = use_prefix_grouper is not None
        use_prefix_grouper = self.use_prefix_grouper if use_prefix_grouper is None else use_prefix_grouper
        if use_prefix_grouper and not capture_hidden:
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
            if capture_hidden:
                step_end_mask = micro_batch["step_end_mask"].bool()
                if step_end_mask.shape != (batch_size, response_length):
                    raise ValueError(
                        "step_end_mask must have shape [batch, response_length], "
                        f"got {tuple(step_end_mask.shape)} instead of {(batch_size, response_length)}."
                    )
                response_attention_mask = attention_mask[:, -response_length:].bool()
                if (step_end_mask & ~response_attention_mask).any():
                    raise ValueError("step_end_mask may select only valid response tokens.")
                full_step_end_mask = torch.zeros(
                    (batch_size, seqlen),
                    dtype=torch.bool,
                    device=step_end_mask.device,
                )
                full_step_end_mask[:, -response_length:] = step_end_mask
                if return_credit_boundary_hidden:
                    prompt_length = seqlen - response_length
                    if prompt_length < 1:
                        raise ValueError("Credit representation requires at least one prompt token.")
                    prompt_attention = attention_mask[:, :prompt_length].bool()
                    prompt_positions = torch.arange(prompt_length, device=attention_mask.device).expand(batch_size, -1)
                    prompt_tail = prompt_positions.masked_fill(~prompt_attention, -1).max(dim=-1).values
                    if torch.any(prompt_tail < 0):
                        raise ValueError("Credit representation requires a nonempty prompt in every response.")
                    prefix_lengths = micro_batch.get("credit_prefix_lengths")
                    if prefix_lengths is None:
                        prefix_lengths = torch.zeros(batch_size, dtype=torch.long, device=attention_mask.device)
                    else:
                        prefix_lengths = prefix_lengths.to(device=attention_mask.device, dtype=torch.long)
                    if prefix_lengths.shape != (batch_size,) or torch.any(prefix_lengths < 0):
                        raise ValueError("credit_prefix_lengths must be one nonnegative integer per response.")
                    if torch.any(prefix_lengths > response_length):
                        raise ValueError("credit_prefix_lengths may not exceed the padded response width.")
                    boundary_positions = prompt_tail.clone()
                    has_fixed_prefix = prefix_lengths > 0
                    boundary_positions[has_fixed_prefix] = prompt_length + prefix_lengths[has_fixed_prefix] - 1
                    boundary_is_active = attention_mask.gather(1, boundary_positions.unsqueeze(-1)).squeeze(-1).bool()
                    if torch.any(~boundary_is_active):
                        raise ValueError("Every credit prefix boundary must select an active context token.")
                    response_positions = torch.arange(response_length, device=attention_mask.device).expand(
                        batch_size, -1
                    )
                    first_step_end = response_positions.masked_fill(~step_end_mask, response_length).min(dim=-1).values
                    if torch.any(first_step_end < prefix_lengths):
                        raise ValueError("Credit step endpoints must occur after the fixed response prefix.")
                    full_step_end_mask[
                        torch.arange(batch_size, device=attention_mask.device),
                        boundary_positions,
                    ] = True
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

                if capture_hidden:
                    step_end_selector = index_first_axis(
                        rearrange(full_step_end_mask.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices,
                    ).squeeze(-1)
                    if is_mask_all_zero:
                        step_end_selector = torch.zeros(
                            input_ids_rmpad.shape,
                            dtype=torch.bool,
                            device=input_ids_rmpad.device,
                        )
                    else:
                        step_end_selector = step_end_selector.unsqueeze(0)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                capture_context = (
                    self._capture_final_hidden_state(step_end_selector) if capture_hidden else nullcontext(None)
                )
                with capture_context as captured_hidden:
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if capture_hidden:
                    step_hidden = self._take_captured_hidden(captured_hidden)

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = not calculate_entropy
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

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                capture_context = (
                    self._capture_final_hidden_state(full_step_end_mask) if capture_hidden else nullcontext(None)
                )
                with capture_context as captured_hidden:
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if capture_hidden:
                    step_hidden = self._take_captured_hidden(captured_hidden)

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(
                        logits,
                        micro_batch["responses"],
                        inplace_backward=not calculate_entropy,
                    )
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
            if return_credit_boundary_hidden:
                outputs["credit_boundary_hidden"] = step_hidden
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

    @staticmethod
    def build_counterfactual_value_transitions(
        boundary_hidden: torch.Tensor,
        step_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align each step with its before/after boundary and endpoint flags."""

        if boundary_hidden.ndim != 2 or step_counts.ndim != 1:
            raise ValueError("boundary_hidden must be rank-2 and step_counts must be rank-1")
        expected_boundaries = int(step_counts.sum().item()) + int(step_counts.numel())
        if boundary_hidden.shape[0] != expected_boundaries:
            raise ValueError(f"Expected {expected_boundaries} prompt/step boundaries, got {boundary_hidden.shape[0]}.")
        before_boundaries = []
        after_boundaries = []
        first_step_masks = []
        last_step_masks = []
        before_indices = []
        after_indices = []
        offset = 0
        for count_tensor in step_counts:
            count = int(count_tensor.item())
            if count < 1:
                raise ValueError("Every response must contain at least one semantic step.")
            boundaries = boundary_hidden[offset : offset + count + 1].detach().float()
            before_boundaries.append(boundaries[:-1])
            after_boundaries.append(boundaries[1:])
            first = torch.zeros(count, dtype=torch.bool, device=boundary_hidden.device)
            last = torch.zeros(count, dtype=torch.bool, device=boundary_hidden.device)
            first[0] = True
            last[-1] = True
            first_step_masks.append(first)
            last_step_masks.append(last)
            response_indices = torch.arange(count, device=boundary_hidden.device)
            before_indices.append(response_indices + offset)
            after_indices.append(response_indices + offset + 1)
            offset += count + 1
        return (
            torch.cat(before_boundaries, dim=0),
            torch.cat(after_boundaries, dim=0),
            torch.cat(first_step_masks, dim=0),
            torch.cat(last_step_masks, dim=0),
            torch.cat(before_indices, dim=0),
            torch.cat(after_indices, dim=0),
        )

    def _sync_counterfactual_credit_gradients(self) -> None:
        if torch.distributed.is_initialized():
            for parameter in self.counterfactual_credit_head.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)

    def compute_counterfactual_credit(self, data: DataProto) -> dict[str, torch.Tensor]:
        """Predict boundary values, replace known values, then fit the detached probe."""

        if not self.has_counterfactual_credit_head:
            raise ValueError("Counterfactual credit requires actor.counterfactual_credit_head.enabled=True.")
        required_keys = (
            "step_end_mask",
            "credit_anchor_mask",
            "credit_anchor_targets",
            "credit_anchor_q_targets",
            "credit_anchor_v_targets",
            "credit_start_value_mask",
            "credit_start_value_train_mask",
            "credit_start_value_targets",
            "credit_terminal_value_mask",
            "credit_terminal_value_targets",
        )
        missing_keys = [key for key in required_keys if key not in data.batch]
        if missing_keys:
            raise ValueError("Counterfactual credit actor batch is missing: " + ", ".join(missing_keys))
        required_meta_info = (
            "global_steps",
            "credit_value_loss_weights",
        )
        missing_meta_info = [key for key in required_meta_info if key not in data.meta_info]
        if missing_meta_info:
            raise ValueError("Counterfactual credit meta_info is missing: " + ", ".join(missing_meta_info))
        global_step = int(data.meta_info["global_steps"])
        value_loss_weights = tuple(float(weight) for weight in data.meta_info["credit_value_loss_weights"])
        predict_all_steps = bool(data.meta_info.get("predict_all_credit_steps", True))
        invalid_value_loss_weights = len(value_loss_weights) != 3 or any(
            not math.isfinite(weight) or weight < 0 for weight in value_loss_weights
        )
        if invalid_value_loss_weights:
            raise ValueError("credit_value_loss_weights must contain three finite nonnegative values")
        if sum(value_loss_weights) <= 0:
            raise ValueError("credit_value_loss_weights must contain at least one positive value")

        self.actor_module.eval()
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            *required_keys,
        ]
        if "credit_prefix_lengths" in data.batch:
            select_keys.append("credit_prefix_lengths")
        data = data.select(batch_keys=select_keys)
        use_dynamic_bsz = bool(data.meta_info["use_dynamic_bsz"])
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data,
                max_token_len=max_token_len,
                dp_group=self._get_micro_batch_sync_group(),
            )
        else:
            micro_batches = data.split(int(data.meta_info["micro_batch_size"]))

        value_batches = []
        mc_value_hidden = []
        mc_value_targets = []
        start_value_hidden = []
        start_value_targets = []
        terminal_value_hidden = []
        terminal_value_targets = []
        anchor_credit_targets = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, "pad_token_id": data.meta_info.get("pad_token_id", 0)}
            step_end_mask = model_inputs["step_end_mask"].bool()
            anchor_mask = model_inputs["credit_anchor_mask"].bool()
            if torch.any(anchor_mask & ~step_end_mask):
                raise ValueError("credit_anchor_mask may select only semantic step endpoints.")
            with torch.no_grad():
                forward_outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=float(data.meta_info["temperature"]),
                    return_credit_boundary_hidden=True,
                )
            step_counts = step_end_mask.sum(dim=-1, dtype=torch.long)
            boundary_hidden = forward_outputs["credit_boundary_hidden"].detach().float()
            before_hidden, after_hidden, first_step_mask, last_step_mask, before_indices, after_indices = (
                self.build_counterfactual_value_transitions(boundary_hidden, step_counts)
            )
            packed_anchor_mask = anchor_mask[step_end_mask]
            selected_credit_targets = model_inputs["credit_anchor_targets"].float()[anchor_mask]
            selected_q_targets = model_inputs["credit_anchor_q_targets"].float()[anchor_mask]
            selected_v_targets = model_inputs["credit_anchor_v_targets"].float()[anchor_mask]
            if torch.any(~torch.isfinite(selected_credit_targets)) or torch.any(selected_credit_targets.abs() > 1.0):
                raise ValueError("Anchor credit targets must be finite and in [-1, 1].")
            for name, targets in (
                ("anchor Q", selected_q_targets),
                ("anchor V", selected_v_targets),
                ("start", model_inputs["credit_start_value_targets"].float()),
                ("terminal", model_inputs["credit_terminal_value_targets"].float()),
            ):
                if torch.any(~torch.isfinite(targets)) or torch.any((targets < 0) | (targets > 1)):
                    raise ValueError(f"{name} value targets must be finite and in [0, 1].")

            start_mask = model_inputs["credit_start_value_mask"].bool()
            start_train_mask = model_inputs["credit_start_value_train_mask"].bool()
            terminal_mask = model_inputs["credit_terminal_value_mask"].bool()
            batch_size = step_end_mask.shape[0]
            if any(mask.shape != (batch_size,) for mask in (start_mask, start_train_mask, terminal_mask)):
                raise ValueError("Start and terminal value masks must contain one entry per response.")
            if torch.any(start_train_mask & ~start_mask):
                raise ValueError("credit_start_value_train_mask must be a subset of credit_start_value_mask.")
            first_step_indices = torch.nonzero(first_step_mask, as_tuple=False).flatten()
            last_step_indices = torch.nonzero(last_step_mask, as_tuple=False).flatten()
            if first_step_indices.numel() != batch_size or last_step_indices.numel() != batch_size:
                raise RuntimeError("Each response must contribute exactly one first and terminal semantic step.")

            # Anchor Q/V labels train both sides of an interior transition.
            # First-step V and terminal-step Q are excluded here because their
            # exact labels are already supplied by the start/terminal sources.
            train_anchor_v = packed_anchor_mask & ~first_step_mask
            train_anchor_q = packed_anchor_mask & ~last_step_mask
            mc_value_hidden.extend((before_hidden[train_anchor_v], after_hidden[train_anchor_q]))
            mc_value_targets.extend(
                (
                    selected_v_targets[~first_step_mask[packed_anchor_mask]],
                    selected_q_targets[~last_step_mask[packed_anchor_mask]],
                )
            )
            start_value_hidden.append(before_hidden[first_step_indices[start_train_mask]])
            start_value_targets.append(model_inputs["credit_start_value_targets"].float()[start_train_mask])
            terminal_value_hidden.append(after_hidden[last_step_indices[terminal_mask]])
            terminal_value_targets.append(model_inputs["credit_terminal_value_targets"].float()[terminal_mask])
            anchor_credit_targets.append(selected_credit_targets)
            value_batches.append(
                {
                    "boundary_hidden": boundary_hidden,
                    "step_counts": step_counts,
                    "step_end_mask": step_end_mask,
                    "packed_anchor_mask": packed_anchor_mask,
                    "before_hidden": before_hidden,
                    "after_hidden": after_hidden,
                    "first_step_indices": first_step_indices,
                    "last_step_indices": last_step_indices,
                    "before_indices": before_indices,
                    "after_indices": after_indices,
                    "start_mask": start_mask,
                    "start_targets": model_inputs["credit_start_value_targets"].float(),
                    "terminal_mask": terminal_mask,
                    "terminal_targets": model_inputs["credit_terminal_value_targets"].float(),
                    "anchor_q_targets": selected_q_targets,
                    "anchor_v_targets": selected_v_targets,
                }
            )

        all_anchor_targets = torch.cat(anchor_credit_targets, dim=0)
        device = value_batches[0]["boundary_hidden"].device
        self._ensure_counterfactual_credit_device(device)

        # Predictions used by the policy are computed before this batch's probe
        # update. Known start, anchor-Q/V, and terminal values then replace the
        # corresponding predictions before adjacent values are differenced.
        dense_predictions = []
        raw_anchor_predictions = []
        self.counterfactual_credit_head.eval()
        with torch.no_grad():
            for value_batch in value_batches:
                packed_anchor_mask = value_batch["packed_anchor_mask"]
                raw_anchor_predictions.append(
                    self.counterfactual_credit_head(value_batch["after_hidden"][packed_anchor_mask]).squeeze(-1)
                    - self.counterfactual_credit_head(value_batch["before_hidden"][packed_anchor_mask]).squeeze(-1)
                )
                endpoint_mask = value_batch["step_end_mask"]
                dense = torch.zeros(endpoint_mask.shape, dtype=torch.float32, device=endpoint_mask.device)
                if predict_all_steps:
                    boundary_values = self.counterfactual_credit_head(value_batch["boundary_hidden"])
                    boundary_values = boundary_values.squeeze(-1).float()
                    before_indices = value_batch["before_indices"]
                    after_indices = value_batch["after_indices"]
                    boundary_values[before_indices[packed_anchor_mask]] = value_batch["anchor_v_targets"]
                    boundary_values[after_indices[packed_anchor_mask]] = value_batch["anchor_q_targets"]
                    start_rows = value_batch["start_mask"]
                    terminal_rows = value_batch["terminal_mask"]
                    start_steps = value_batch["first_step_indices"][start_rows]
                    terminal_steps = value_batch["last_step_indices"][terminal_rows]
                    boundary_values[before_indices[start_steps]] = value_batch["start_targets"][start_rows]
                    boundary_values[after_indices[terminal_steps]] = value_batch["terminal_targets"][terminal_rows]
                    dense[endpoint_mask] = boundary_values[after_indices] - boundary_values[before_indices]
                dense_predictions.append(dense)

        all_raw_anchor_predictions = torch.cat(raw_anchor_predictions, dim=0)

        self.counterfactual_credit_head.train()
        should_update = self.counterfactual_credit_last_global_step != global_step
        # Weight each sign decision by the amount of credit at stake. These are
        # raw value-probe differences, before exact boundary values are inserted.
        target_magnitudes = all_anchor_targets.abs()
        correct_directions = all_raw_anchor_predictions * all_anchor_targets > 0
        direction_statistics = torch.stack(
            (
                (target_magnitudes * correct_directions).sum(),
                target_magnitudes.sum(),
            )
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(direction_statistics, op=torch.distributed.ReduceOp.SUM)
        direction_agreement = torch.where(
            direction_statistics[1] > 0,
            direction_statistics[0] / direction_statistics[1],
            direction_statistics.new_tensor(float("nan")),
        )
        source_hidden = (
            torch.cat(mc_value_hidden, dim=0),
            torch.cat(start_value_hidden, dim=0),
            torch.cat(terminal_value_hidden, dim=0),
        )
        source_targets = (
            torch.cat(mc_value_targets, dim=0),
            torch.cat(start_value_targets, dim=0),
            torch.cat(terminal_value_targets, dim=0),
        )
        global_source_counts = torch.tensor(
            [targets.numel() for targets in source_targets],
            dtype=torch.float32,
            device=device,
        )
        global_difference_count = torch.tensor(float(all_anchor_targets.numel()), dtype=torch.float32, device=device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_source_counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(global_difference_count, op=torch.distributed.ReduceOp.SUM)
        configured_weights = torch.tensor(value_loss_weights, dtype=torch.float32, device=device)
        active_weights = configured_weights * (global_source_counts > 0)
        active_weights = active_weights / active_weights.sum().clamp_min(torch.finfo(torch.float32).eps)

        head_config = self.config.get("counterfactual_credit_head")
        difference_loss_weight = float(head_config.get("difference_loss_weight", 0.25))
        optimizer_steps_per_batch = int(head_config.get("optimizer_steps_per_batch", 2))

        def compute_probe_losses():
            source_loss_sums = []
            source_absolute_error_sums = []
            for hidden, targets in zip(source_hidden, source_targets, strict=True):
                if targets.numel() > 0:
                    value_predictions = self.counterfactual_credit_head(hidden).squeeze(-1)
                    source_loss_sums.append(
                        torch.nn.functional.binary_cross_entropy(value_predictions, targets, reduction="sum")
                    )
                    source_absolute_error_sums.append((value_predictions.detach() - targets).abs().sum())
                else:
                    source_loss_sums.append(torch.zeros((), dtype=torch.float32, device=device))
                    source_absolute_error_sums.append(torch.zeros((), dtype=torch.float32, device=device))

            difference_predictions = []
            for value_batch in value_batches:
                packed_anchor_mask = value_batch["packed_anchor_mask"]
                difference_predictions.append(
                    self.counterfactual_credit_head(value_batch["after_hidden"][packed_anchor_mask]).squeeze(-1)
                    - self.counterfactual_credit_head(value_batch["before_hidden"][packed_anchor_mask]).squeeze(-1)
                )
            all_difference_predictions = torch.cat(difference_predictions, dim=0)
            difference_loss_sum = torch.nn.functional.smooth_l1_loss(
                all_difference_predictions,
                all_anchor_targets,
                reduction="sum",
            )
            return torch.stack(source_loss_sums), torch.stack(source_absolute_error_sums), difference_loss_sum

        source_loss_sums, source_absolute_error_sums, difference_loss_sum = compute_probe_losses()
        metric_source_loss_sums = source_loss_sums.detach().clone()
        metric_source_absolute_error_sums = source_absolute_error_sums.clone()
        metric_difference_loss_sum = difference_loss_sum.detach().clone()
        if should_update and (global_source_counts.sum() + global_difference_count).item() > 0:
            for optimizer_step in range(optimizer_steps_per_batch):
                if optimizer_step > 0:
                    source_loss_sums, _, difference_loss_sum = compute_probe_losses()
                self.counterfactual_credit_optimizer.zero_grad(set_to_none=True)
                value_bce = (active_weights * source_loss_sums / global_source_counts.clamp_min(1.0)).sum()
                difference_smooth_l1 = difference_loss_sum / global_difference_count.clamp_min(1.0)
                objective = sum(parameter.sum() * 0.0 for parameter in self.counterfactual_credit_head.parameters())
                objective = objective + value_bce + difference_loss_weight * difference_smooth_l1
                objective.backward()
                self._sync_counterfactual_credit_gradients()
                self.counterfactual_credit_optimizer.step()
                self.counterfactual_credit_updates += 1
            self.counterfactual_credit_last_global_step = global_step
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(metric_source_loss_sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(metric_source_absolute_error_sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(metric_difference_loss_sum, op=torch.distributed.ReduceOp.SUM)
        self.counterfactual_credit_head.eval()
        predictions = torch.cat(dense_predictions, dim=0)
        if use_dynamic_bsz:
            predictions = restore_dynamic_batch(predictions, batch_idx_list)
        batch_size = predictions.shape[0]
        mean_value_bce = (
            active_weights * metric_source_loss_sums / global_source_counts.clamp_min(1.0)
        ).sum()
        mean_absolute_error = (
            active_weights * metric_source_absolute_error_sums / global_source_counts.clamp_min(1.0)
        ).sum()
        mean_difference_smooth_l1 = metric_difference_loss_sum / global_difference_count.clamp_min(1.0)
        mean_total_loss = mean_value_bce + difference_loss_weight * mean_difference_smooth_l1
        return {
            "credit_predictions": predictions,
            "credit_head_value_bce": mean_value_bce.expand(batch_size).clone(),
            "credit_head_difference_smooth_l1": mean_difference_smooth_l1.expand(batch_size).clone(),
            "credit_head_total_loss": mean_total_loss.expand(batch_size).clone(),
            "credit_head_value_mae": mean_absolute_error.expand(batch_size).clone(),
            "credit_head_direction_agreement": direction_agreement.expand(batch_size).clone(),
        }

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self,
        data: DataProto,
        calculate_entropy: bool = False,
    ) -> dict[str, torch.Tensor | float]:
        """Compute response log probabilities and optional actor diagnostics."""

        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [key for key in ("prompts", "response_mask") if key in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data,
                max_token_len=max_token_len,
                dp_group=self._get_micro_batch_sync_group(),
            )
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
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
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        """Update the actor policy on PPO mini-batches."""

        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        dgpo_vocab_size = self._model_vocab_size(self.actor_module) if loss_mode == "dgpo" else None
        kl_loss_needs_ref = self.config.use_kl_loss and float(self.config.kl_loss_coef) != 0.0

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch:
            select_keys.append("prompts")
        if kl_loss_needs_ref or loss_mode == "dgpo":
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch:
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch:
            select_keys.append("rollout_log_probs")

        non_tensor_select_keys = []
        if "multi_modal_inputs" in data.non_tensor_batch:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch:
            non_tensor_select_keys.append("uid")

        data = data.select(
            batch_keys=select_keys,
            non_tensor_batch_keys=non_tensor_select_keys,
        )
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch,
                        max_token_len=max_token_len,
                        dp_group=self._get_micro_batch_sync_group(),
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
                    model_inputs = {
                        **micro_batch.batch,
                        **micro_batch.non_tensor_batch,
                        "pad_token_id": pad_token_id,
                    }
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = (
                        self.config.calculate_entropy or entropy_coeff != 0 or loss_mode in {"my", "dgpo"}
                    )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    if getattr(self.config, "use_rollout_log_probs", False):
                        old_log_prob = model_inputs["old_log_probs"]
                    elif on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    rollout_is_weights = model_inputs.get("rollout_is_weights")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    policy_loss_kwargs = {
                        "old_log_prob": old_log_prob,
                        "log_prob": log_prob,
                        "advantages": advantages,
                        "response_mask": response_mask,
                        "loss_agg_mode": loss_agg_mode,
                        "config": self.config,
                        "rollout_is_weights": rollout_is_weights,
                    }
                    if loss_mode == "dgpo":
                        policy_loss_kwargs.update(
                            ref_log_prob=model_inputs["ref_log_prob"],
                            entropy=entropy,
                            vocab_size=dgpo_vocab_size,
                        )
                    elif loss_mode == "my":
                        policy_loss_kwargs["entropy"] = entropy

                    policy_loss_result = policy_loss_fn(**policy_loss_kwargs)
                    if loss_mode == "my":
                        pg_loss, pg_metrics, _ = policy_loss_result
                    else:
                        pg_loss, pg_metrics = policy_loss_result
                    micro_batch_metrics.update(pg_metrics)

                    rollout_log_prob = model_inputs.get("rollout_log_probs")
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import (
                            compute_rollout_corr_metrics_from_logprobs,
                        )

                        micro_batch_metrics.update(
                            compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                        )

                    policy_loss = pg_loss
                    if entropy is not None:
                        entropy_agg = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if kl_loss_needs_ref:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss += kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
