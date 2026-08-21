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
from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint as activation_checkpoint

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

__all__ = [
    "DataParallelPPOActor",
    "StepValueProbe",
    "dapo_reference_topk_forward_kl",
    "gather_dapo_reference_teacher_topk",
    "mix_dapo_reference_kl_loss",
    "summarize_dapo_reference_teacher_topk",
]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_DAPO_REFERENCE_KL_TOKEN_CHUNK_SIZE = 512


def summarize_dapo_reference_teacher_topk(
    teacher_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress a teacher distribution to teacher top-k log-probs and tail mass."""

    if teacher_logits.ndim != 2:
        raise ValueError(
            f"DAPO reference top-k teacher logits must have shape [tokens, vocab], got {tuple(teacher_logits.shape)}."
        )
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"DAPO reference top_k must be a positive integer, got {top_k!r}.")
    vocab_size = teacher_logits.shape[-1]
    if vocab_size < 1:
        raise ValueError("DAPO reference top-k teacher logits must have a non-empty vocabulary dimension.")

    top_k = min(top_k, vocab_size)
    teacher_logits_f = teacher_logits.detach().float()
    top_values, top_indices = torch.topk(teacher_logits_f, k=top_k, dim=-1)
    teacher_log_z = torch.logsumexp(teacher_logits_f, dim=-1, keepdim=True)
    teacher_top_log_probs = top_values - teacher_log_z
    if top_k == vocab_size:
        teacher_tail_prob = torch.zeros_like(teacher_log_z.squeeze(-1))
    else:
        teacher_top_log_mass = torch.logsumexp(teacher_top_log_probs, dim=-1).clamp_max(0.0)
        teacher_tail_prob = (-torch.expm1(teacher_top_log_mass)).clamp_min(torch.finfo(torch.float32).tiny)
    return top_indices, teacher_top_log_probs, teacher_tail_prob


def _dapo_reference_topk_forward_kl_chunk(
    student_logits: torch.Tensor,
    teacher_top_indices: torch.Tensor,
    teacher_top_log_probs: torch.Tensor,
    teacher_tail_prob: torch.Tensor,
) -> torch.Tensor:
    """Compute one token chunk of the teacher-top-k-plus-tail forward KL."""

    student_logits_f = student_logits.float()
    student_log_z = torch.logsumexp(student_logits_f, dim=-1, keepdim=True)
    student_top_log_probs = student_logits_f.gather(-1, teacher_top_indices) - student_log_z
    teacher_top_prob = teacher_top_log_probs.exp()
    token_kl = (teacher_top_prob * (teacher_top_log_probs - student_top_log_probs)).sum(dim=-1)

    if teacher_top_indices.shape[-1] < student_logits.shape[-1]:
        student_top_log_mass = torch.logsumexp(student_top_log_probs, dim=-1).clamp_max(0.0)
        student_tail_prob = (-torch.expm1(student_top_log_mass)).clamp_min(torch.finfo(torch.float32).tiny)
        token_kl = token_kl + teacher_tail_prob * (teacher_tail_prob.log() - student_tail_prob.log())
    return token_kl


def dapo_reference_topk_forward_kl(
    student_logits: torch.Tensor,
    teacher_top_indices: torch.Tensor,
    teacher_top_log_probs: torch.Tensor,
    teacher_tail_prob: torch.Tensor,
    token_chunk_size: int = _DAPO_REFERENCE_KL_TOKEN_CHUNK_SIZE,
) -> torch.Tensor:
    """Return a memory-bounded teacher-top-k-plus-tail approximation of ``KL(teacher || student)``.

    Each token chunk is activation-checkpointed while gradients are enabled. This prevents the
    full-vocabulary FP32 logits and their gradients from being resident for all selected response
    tokens at once, without changing the KL approximation.
    """

    if student_logits.ndim != 2:
        raise ValueError(
            f"DAPO reference top-k student logits must have shape [tokens, vocab], got {tuple(student_logits.shape)}."
        )
    expected_shape = teacher_top_indices.shape
    if teacher_top_log_probs.shape != expected_shape:
        raise ValueError("DAPO reference teacher top-k indices and log-probs must have matching shapes.")
    if teacher_tail_prob.shape != expected_shape[:-1]:
        raise ValueError("DAPO reference teacher tail probability must have shape [tokens].")
    if student_logits.shape[0] != expected_shape[0]:
        raise ValueError("DAPO reference teacher and student top-k summaries must have the same token count.")
    if not isinstance(token_chunk_size, int) or isinstance(token_chunk_size, bool) or token_chunk_size < 1:
        raise ValueError(f"DAPO reference KL token_chunk_size must be a positive integer, got {token_chunk_size!r}.")

    num_tokens = student_logits.shape[0]
    if num_tokens == 0:
        return _dapo_reference_topk_forward_kl_chunk(
            student_logits,
            teacher_top_indices,
            teacher_top_log_probs,
            teacher_tail_prob,
        )

    checkpoint_chunks = torch.is_grad_enabled() and student_logits.requires_grad
    token_kl_chunks = []
    for start in range(0, num_tokens, token_chunk_size):
        stop = min(start + token_chunk_size, num_tokens)
        chunk_args = (
            student_logits[start:stop],
            teacher_top_indices[start:stop],
            teacher_top_log_probs[start:stop],
            teacher_tail_prob[start:stop],
        )
        if checkpoint_chunks:
            chunk_kl = activation_checkpoint(
                _dapo_reference_topk_forward_kl_chunk,
                *chunk_args,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            chunk_kl = _dapo_reference_topk_forward_kl_chunk(*chunk_args)
        token_kl_chunks.append(chunk_kl)
    return torch.cat(token_kl_chunks, dim=0)


def gather_dapo_reference_teacher_topk(
    teacher_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    row_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather cached old-policy teacher targets in micro-batch token order."""

    if row_ids.ndim != 1:
        raise ValueError(f"DAPO reference KL row_ids must be one-dimensional, got {tuple(row_ids.shape)}.")
    if loss_mask.ndim != 2 or loss_mask.shape[0] != row_ids.shape[0]:
        raise ValueError(
            "DAPO reference KL loss_mask must have shape [batch, response_length] with the same "
            f"batch size as row_ids, got {tuple(loss_mask.shape)} and {tuple(row_ids.shape)}."
        )

    selected_rows = loss_mask.bool().any(dim=-1)
    selected_row_ids = row_ids[selected_rows].detach().cpu().tolist()
    selected_token_counts = loss_mask[selected_rows].sum(dim=-1).detach().cpu().tolist()
    cached_indices = []
    cached_log_probs = []
    cached_tail_probs = []
    top_k = None
    for row_id, token_count in zip(selected_row_ids, selected_token_counts, strict=True):
        row_id = int(row_id)
        token_count = int(token_count)
        if row_id not in teacher_cache:
            raise ValueError(f"DAPO reference KL is missing an old-policy teacher cache entry for row {row_id}.")
        indices, log_probs, tail_probs = teacher_cache[row_id]
        if indices.ndim != 2 or log_probs.shape != indices.shape or tail_probs.shape != indices.shape[:-1]:
            raise ValueError(f"DAPO reference KL teacher cache entry for row {row_id} has invalid shapes.")
        if indices.shape[0] != token_count:
            raise ValueError(
                f"DAPO reference KL teacher cache row {row_id} contains {indices.shape[0]} tokens, "
                f"but its current loss mask selects {token_count}."
            )
        if top_k is None:
            top_k = indices.shape[-1]
        elif indices.shape[-1] != top_k:
            raise ValueError("DAPO reference KL teacher cache entries must use one shared top_k.")
        cached_indices.append(indices)
        cached_log_probs.append(log_probs)
        cached_tail_probs.append(tail_probs)

    if not cached_indices:
        raise ValueError("DAPO reference KL cannot gather an empty old-policy teacher target.")
    return (
        torch.cat(cached_indices, dim=0).to(device=device, dtype=torch.long),
        torch.cat(cached_log_probs, dim=0).to(device=device),
        torch.cat(cached_tail_probs, dim=0).to(device=device),
    )


def mix_dapo_reference_kl_loss(
    dapo_loss: torch.Tensor,
    reference_kl_loss: torch.Tensor,
    loss_coef: float,
) -> torch.Tensor:
    """Return the configured convex combination of DAPO and reference-KL losses."""

    return (1.0 - loss_coef) * dapo_loss + loss_coef * reference_kl_loss


class StepValueProbe(nn.Module):
    """Two-layer MLP that predicts eventual success from a step-tail hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim, dtype=torch.float32)
        self.activation = nn.SiLU()
        self.output_layer = nn.Linear(self.hidden_dim, 1, dtype=torch.float32)

    def reset_parameters(self) -> None:
        """Reset both trainable layers without changing the architecture."""
        self.input_layer.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.activation(self.input_layer(hidden)))


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

        self._init_step_value_probe()

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
        raise ValueError("step_value_probe could not infer the actor hidden size from the model config.")

    def _init_step_value_probe(self) -> None:
        probe_config = self.config.get("step_value_probe", None)
        self.step_value_probe_enabled = bool(probe_config is not None and probe_config.get("enabled", False))
        self.step_value_probe = None
        self.step_value_probe_optimizer = None
        self.step_value_probe_updates = 0
        self.step_value_probe_warmup_completed_at = None
        self.step_value_probe_last_global_step = None
        self._step_value_probe_needs_broadcast = False

        if not self.step_value_probe_enabled:
            return
        if self.actor_optimizer is None:
            raise ValueError("step_value_probe is supported only by the trainable actor.")
        if self.use_ulysses_sp:
            raise ValueError("step_value_probe does not support Ulysses sequence parallelism (SP > 1).")
        if self.use_prefix_grouper:
            raise ValueError("step_value_probe does not support use_prefix_grouper=True.")
        if self.use_fused_kernels:
            raise ValueError("step_value_probe does not support use_fused_kernels=True.")

        model = self._unwrap_model(self.actor_module)
        model_config = getattr(model, "config", None)
        if getattr(model_config, "vision_config", None) is not None:
            raise ValueError("step_value_probe currently supports text-only causal language models.")
        output_embeddings = getattr(model, "get_output_embeddings", lambda: None)()
        if not isinstance(output_embeddings, nn.Module):
            raise ValueError(
                "step_value_probe requires a causal language model with a module returned by get_output_embeddings()."
            )

        hidden_size = self._model_hidden_size(model)
        probe_hidden_size = int(probe_config.get("hidden_dim", 256))
        # The auxiliary probe must not perturb the actor's checkpointed RNG stream.
        with torch.random.fork_rng(devices=[], device_type="cpu"):
            self.step_value_probe = StepValueProbe(hidden_size, probe_hidden_size)
        self._step_value_probe_needs_broadcast = True
        self.step_value_probe_optimizer = torch.optim.AdamW(
            self.step_value_probe.parameters(),
            lr=float(probe_config.get("lr", 1e-3)),
            weight_decay=float(probe_config.get("weight_decay", 0.0)),
        )

    @property
    def has_step_value_probe(self) -> bool:
        return self.step_value_probe is not None

    def _broadcast_step_value_probe(self) -> bool:
        if not self.has_step_value_probe or not torch.distributed.is_initialized():
            return False
        for parameter in self.step_value_probe.parameters():
            torch.distributed.broadcast(parameter.data, src=0)
        return True

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

    def _ensure_step_value_probe_device(self, device: torch.device) -> None:
        if not self.has_step_value_probe:
            return
        self.step_value_probe.to(device=device, dtype=torch.float32)
        for state in self.step_value_probe_optimizer.state.values():
            for key, value in list(state.items()):
                state[key] = self._move_optimizer_value(value, device)
        if self._step_value_probe_needs_broadcast and self._broadcast_step_value_probe():
            self._step_value_probe_needs_broadcast = False

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

    def step_value_probe_state_dict(self) -> dict | None:
        if not self.has_step_value_probe:
            return None
        return {
            "format_version": 2,
            "architecture": {
                "type": "two_layer_mlp",
                "input_dim": self.step_value_probe.input_dim,
                "hidden_dim": self.step_value_probe.hidden_dim,
            },
            "model": {name: tensor.detach().cpu() for name, tensor in self.step_value_probe.state_dict().items()},
            "optimizer": self._nested_to_cpu(self.step_value_probe_optimizer.state_dict()),
            "updates": self.step_value_probe_updates,
            "warmup_completed_at": self.step_value_probe_warmup_completed_at,
            "last_global_step": self.step_value_probe_last_global_step,
        }

    @torch.no_grad()
    def reset_step_value_probe(self) -> None:
        """Reset an enabled probe and require a fresh warmup round."""
        if not self.has_step_value_probe:
            return
        probe_device = next(self.step_value_probe.parameters()).device
        fork_devices = [] if probe_device.type == "cpu" else [probe_device]
        with torch.random.fork_rng(devices=fork_devices, device_type=probe_device.type):
            self.step_value_probe.reset_parameters()
        self._step_value_probe_needs_broadcast = True
        self.step_value_probe_optimizer.state.clear()
        self.step_value_probe_updates = 0
        self.step_value_probe_warmup_completed_at = None
        self.step_value_probe_last_global_step = None

    def load_step_value_probe_state_dict(self, state_dict: dict | None) -> None:
        if not self.has_step_value_probe:
            return
        if state_dict is None:
            self.reset_step_value_probe()
            return
        format_version = int(state_dict.get("format_version", 0))
        if format_version != 2:
            raise RuntimeError("Unsupported step-value probe checkpoint format.")
        expected_architecture = {
            "type": "two_layer_mlp",
            "input_dim": self.step_value_probe.input_dim,
            "hidden_dim": self.step_value_probe.hidden_dim,
        }
        checkpoint_architecture = state_dict.get("architecture")
        if checkpoint_architecture != expected_architecture:
            raise RuntimeError(
                "Step-value probe checkpoint architecture does not match the configured probe: "
                f"checkpoint={checkpoint_architecture!r}, configured={expected_architecture!r}."
            )
        self.step_value_probe.load_state_dict(state_dict["model"], strict=True)
        self.step_value_probe_optimizer.load_state_dict(state_dict["optimizer"])
        self.step_value_probe_updates = int(state_dict.get("updates", 0))
        self.step_value_probe_warmup_completed_at = state_dict.get("warmup_completed_at")
        self.step_value_probe_last_global_step = state_dict.get("last_global_step")
        self._step_value_probe_needs_broadcast = False

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
        return_topk_logits: bool = False,
        return_full_logits: bool = False,
        return_valid_logits: bool = False,
        valid_logits_only: bool = False,
        return_step_hidden: bool = False,
        return_step_value_hidden: bool | None = None,
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
                if return_topk_logits is True:
                    logits: (bs, response_len, min(100, vocab_size))
                    logits_indices: (bs, response_len, min(100, vocab_size))
                if return_full_logits is True:
                    full_logits: (bs, response_len, vocab_size)
                if return_valid_logits is True:
                    valid_logits: (num_valid_response_tokens, vocab_size)
                    if valid_logits_only is True, no policy outputs are computed
                if return_step_hidden is True:
                    step_hidden: hidden states selected by ``step_end_mask`` in
                        row-major order; this does not enable or invoke a probe.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False) and not (
            return_full_logits or return_valid_logits
        )
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_remove_padding = self.use_remove_padding if use_remove_padding is None else use_remove_padding
        legacy_step_hidden_requested = return_step_value_hidden is not None
        if legacy_step_hidden_requested:
            if return_step_hidden and not bool(return_step_value_hidden):
                raise ValueError("Conflicting return_step_hidden and return_step_value_hidden settings")
            return_step_hidden = bool(return_step_value_hidden)
        if return_step_hidden:
            if self.use_ulysses_sp:
                raise ValueError("Step-hidden capture does not support Ulysses sequence parallelism (SP > 1).")
            if self.use_fused_kernels:
                raise ValueError("Step-hidden capture does not support use_fused_kernels=True.")
            if self.use_prefix_grouper:
                raise ValueError("Step-hidden capture does not support use_prefix_grouper=True.")
            if return_topk_logits:
                raise ValueError("Step-hidden capture cannot be combined with return_topk_logits=True.")
            if "step_end_mask" not in micro_batch:
                raise ValueError("Step-hidden capture requires step_end_mask in the actor batch.")
        if return_topk_logits and self.use_fused_kernels:
            raise ValueError("DGPO top-k logits are not supported with use_fused_kernels=True.")
        if valid_logits_only and not return_valid_logits:
            raise ValueError("valid_logits_only=True requires return_valid_logits=True.")
        if return_full_logits or return_valid_logits:
            if self.use_fused_kernels:
                raise ValueError("Full-vocabulary logits do not support use_fused_kernels=True.")
            if self.use_ulysses_sp:
                raise ValueError("Full-vocabulary logits do not support Ulysses sequence parallelism.")
            if return_full_logits and return_valid_logits:
                raise ValueError("return_full_logits and return_valid_logits cannot both be enabled.")
            if return_full_logits and (return_topk_logits or return_step_hidden or calculate_entropy):
                raise ValueError("return_full_logits cannot be combined with other forward outputs.")
        # PrefixGrouper path for shared-prefix optimization
        prefix_grouper_override = use_prefix_grouper is not None
        use_prefix_grouper = self.use_prefix_grouper if use_prefix_grouper is None else use_prefix_grouper
        if use_prefix_grouper and not return_topk_logits and not return_full_logits and not return_valid_logits:
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
            valid_logits = None
            if return_step_hidden:
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

                if return_step_hidden:
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
                    self._capture_final_hidden_state(step_end_selector) if return_step_hidden else nullcontext(None)
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

                if return_step_hidden:
                    step_hidden = self._take_captured_hidden(captured_hidden)

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    if return_full_logits or return_valid_logits:
                        valid_response_mask = micro_batch.get(
                            "dapo_reference_kl_loss_mask", micro_batch["response_mask"]
                        )
                        predictor_mask = torch.zeros((batch_size, seqlen), dtype=torch.bool, device=input_ids.device)
                        predictor_mask[:, -response_length - 1 : -1] = valid_response_mask.bool()
                        predictor_selector = index_first_axis(
                            rearrange(predictor_mask.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                        ).squeeze(-1)
                        selected_logits = logits_rmpad[predictor_selector]
                        if return_valid_logits:
                            valid_logits = selected_logits
                            if valid_logits_only:
                                return {"valid_logits": valid_logits}
                        else:
                            response_selector = micro_batch["response_mask"].bool().reshape(-1).nonzero().squeeze(-1)
                            flat_response_logits = logits_rmpad.new_zeros(
                                (batch_size * response_length, logits_rmpad.shape[-1])
                            )
                            flat_response_logits = flat_response_logits.index_copy(
                                0, response_selector, selected_logits
                            )
                            response_logits = flat_response_logits.view(
                                batch_size, response_length, logits_rmpad.shape[-1]
                            )
                            return {"full_logits": response_logits}
                    if return_topk_logits:
                        topk = min(100, logits_rmpad.size(-1))
                        topk_logits, topk_logits_indices = torch.topk(logits_rmpad.detach(), k=topk, dim=-1)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = not (calculate_entropy or return_valid_logits)
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

                    if return_topk_logits:
                        topk_logits = gather_outputs_and_unpad(
                            topk_logits,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        topk_logits_indices = gather_outputs_and_unpad(
                            topk_logits_indices,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if return_topk_logits:
                        topk_logits = topk_logits[:0]
                        topk_logits_indices = topk_logits_indices[:0]

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

                if return_topk_logits:
                    full_topk_logits = pad_input(
                        hidden_states=topk_logits,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_logits_indices = pad_input(
                        hidden_states=topk_logits_indices,
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
                if return_topk_logits:
                    topk_logits = full_topk_logits[:, -response_length - 1 : -1, :]
                    topk_logits_indices = full_topk_logits_indices[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                capture_context = (
                    self._capture_final_hidden_state(full_step_end_mask) if return_step_hidden else nullcontext(None)
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

                if return_step_hidden:
                    step_hidden = self._take_captured_hidden(captured_hidden)

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if return_valid_logits:
                        valid_response_mask = micro_batch.get(
                            "dapo_reference_kl_loss_mask", micro_batch["response_mask"]
                        )
                        valid_logits = logits[valid_response_mask.bool()]
                        if valid_logits_only:
                            return {"valid_logits": valid_logits}
                    if return_full_logits:
                        return {"full_logits": logits}
                    if return_topk_logits:
                        topk = min(100, logits.size(-1))
                        topk_logits, topk_logits_indices = torch.topk(logits.detach(), k=topk, dim=-1)
                    log_probs = logprobs_from_logits(
                        logits,
                        micro_batch["responses"],
                        inplace_backward=not return_valid_logits,
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
            if return_topk_logits:
                outputs["logits"] = topk_logits
                outputs["logits_indices"] = topk_logits_indices
            if return_valid_logits:
                outputs["valid_logits"] = valid_logits
            if return_step_hidden:
                outputs["step_hidden"] = step_hidden
                if legacy_step_hidden_requested:
                    outputs["step_value_hidden"] = step_hidden
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

    def _step_value_is_ready(self, global_step: int) -> bool:
        probe_config = self.config.step_value_probe
        return (
            self.step_value_probe_updates >= int(probe_config.warmup_updates)
            and self.step_value_probe_warmup_completed_at is not None
            and global_step > int(self.step_value_probe_warmup_completed_at)
        )

    def _sync_step_value_probe_gradients(self, local_trajectory_count: int) -> int:
        probe_parameter = next(self.step_value_probe.parameters())
        count = torch.tensor(
            float(local_trajectory_count),
            dtype=torch.float32,
            device=probe_parameter.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            for parameter in self.step_value_probe.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)

        global_trajectory_count = int(count.item())
        if global_trajectory_count > 0:
            for parameter in self.step_value_probe.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(count)
        return global_trajectory_count

    def _step_value_probe_grad_norm(self) -> torch.Tensor:
        """Return the L2 norm of the synchronized probe gradients."""
        probe_parameter = next(self.step_value_probe.parameters())
        squared_norm = torch.zeros((), dtype=torch.float32, device=probe_parameter.device)
        for parameter in self.step_value_probe.parameters():
            if parameter.grad is not None:
                squared_norm.add_(parameter.grad.detach().float().square().sum())
        return squared_norm.sqrt()

    def _update_step_value_probe(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        endpoint_weights: torch.Tensor,
        local_trajectory_count: int,
        global_step: int,
    ) -> torch.Tensor:
        """Update the replicated probe and return its synchronized gradient norm."""
        if self.step_value_probe_last_global_step is not None:
            previous_step = int(self.step_value_probe_last_global_step)
            if global_step < previous_step:
                raise ValueError(
                    "step_value_probe received a global_steps value older than its last update: "
                    f"{global_step} < {previous_step}."
                )
            if global_step == previous_step:
                probe_parameter = next(self.step_value_probe.parameters())
                gradient_norm = torch.zeros((), dtype=torch.float32, device=probe_parameter.device)
                return gradient_norm

        probe_config = self.config.step_value_probe
        is_warmup = self.step_value_probe_updates < int(probe_config.warmup_updates)
        epochs = int(probe_config.warmup_epochs if is_warmup else probe_config.update_epochs)
        self.step_value_probe.train()
        self._ensure_step_value_probe_device(hidden.device)

        did_update = False
        gradient_norm = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for _ in range(epochs):
            self.step_value_probe_optimizer.zero_grad(set_to_none=True)
            if hidden.numel() > 0:
                logits = self.step_value_probe(hidden.float()).squeeze(-1)
                endpoint_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    reduction="none",
                )
                (endpoint_losses * endpoint_weights).sum().backward()

            global_trajectory_count = self._sync_step_value_probe_gradients(local_trajectory_count)
            gradient_norm = self._step_value_probe_grad_norm()
            if global_trajectory_count > 0:
                self.step_value_probe_optimizer.step()
                did_update = True

        if did_update:
            self.step_value_probe_updates += 1
            self.step_value_probe_last_global_step = global_step
            if self.step_value_probe_updates == int(probe_config.warmup_updates):
                self.step_value_probe_warmup_completed_at = global_step
        self.step_value_probe.eval()
        return gradient_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self,
        data: DataProto,
        calculate_entropy: bool = False,
        return_topk_logits: bool = False,
        compute_step_value_probe: bool = False,
        compute_similarity_step_embeddings: bool = False,
        compute_step_values: bool | None = None,
    ) -> dict[str, torch.Tensor | float]:
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
                - ``logits``: top-k logits of shape [batch_size, response_length, k], when requested.
                - ``logits_indices``: top-k token indices with the same shape as ``logits``, when requested.
                - ``step_values``: sparse step-end probabilities with shape [batch_size, response_length].
                - ``step_value_trajectory_logit_mean``: trajectory-balanced input for prompt-center calibration.
                - ``step_value_ready``: whether the pre-update predictions are usable for policy training.
                - ``step_value_probe_loss``: pre-update mean probe loss for each trajectory.
                - ``step_value_probe_grad_norm``: synchronized gradient L2 norm of the final probe epoch.
                - ``step_value_audit_trajectory_logit_mean``: post-update mean raw logit for audit trajectories.
                - ``step_value_audit_endpoint_count``: number of cached endpoints used by each audit trajectory.
                - ``step_value_audit_ready_next``: whether an audit trajectory is ready for next-step calibration.
                - ``step_value_forward_row_id``: stable row identifier echoed through dynamic batching.
                - ``similarity_step_embeddings``: packed normalized ``H(end)-H(start)`` step representations.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # ``compute_step_values`` is the deprecated name for the built-in
        # probe request. Keep it for out-of-tree callers without conflating it
        # with the provider-agnostic step-value estimator.
        if compute_step_values is not None:
            if compute_step_value_probe and not bool(compute_step_values):
                raise ValueError("Conflicting compute_step_value_probe and compute_step_values settings.")
            compute_step_value_probe = bool(compute_step_values)

        if compute_step_value_probe:
            if not self.has_step_value_probe:
                raise ValueError("compute_step_value_probe requires actor.step_value_probe.enabled=True.")
            if return_topk_logits:
                raise ValueError("step_value_probe cannot be combined with return_topk_logits=True.")
            missing_keys = [key for key in ("step_end_mask", "step_value_targets") if key not in data.batch]
            if missing_keys:
                raise ValueError("compute_step_value_probe requires actor batch keys: " + ", ".join(missing_keys))
            if "global_steps" not in data.meta_info:
                raise ValueError("compute_step_value_probe requires global_steps in data.meta_info.")
            global_step = int(data.meta_info["global_steps"])
            step_value_ready_before_update = self._step_value_is_ready(global_step)

            audit_contract_keys = (
                "step_value_probe_update_mask",
                "step_value_prompt_center_audit_mask",
                "step_value_forward_row_id",
            )
            present_audit_contract_keys = [key for key in audit_contract_keys if key in data.batch]
            delayed_audit_requested = bool(present_audit_contract_keys)
            if delayed_audit_requested and len(present_audit_contract_keys) != len(audit_contract_keys):
                missing_audit_contract_keys = [
                    key for key in audit_contract_keys if key not in present_audit_contract_keys
                ]
                raise ValueError(
                    "Delayed step-value audit requires all actor batch keys: "
                    + ", ".join(missing_audit_contract_keys)
                )
            if delayed_audit_requested:
                batch_size = data.batch["responses"].shape[0]
                update_mask = data.batch["step_value_probe_update_mask"]
                audit_mask = data.batch["step_value_prompt_center_audit_mask"]
                forward_row_id = data.batch["step_value_forward_row_id"]
                for key, mask in (
                    ("step_value_probe_update_mask", update_mask),
                    ("step_value_prompt_center_audit_mask", audit_mask),
                ):
                    if mask.ndim != 1 or mask.shape[0] != batch_size or mask.dtype != torch.bool:
                        raise ValueError(f"{key} must be a bool tensor with shape [batch_size].")
                integer_dtypes = {
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                }
                if (
                    forward_row_id.ndim != 1
                    or forward_row_id.shape[0] != batch_size
                    or forward_row_id.dtype not in integer_dtypes
                ):
                    raise ValueError("step_value_forward_row_id must be an integer tensor with shape [batch_size].")
                if torch.unique(forward_row_id).numel() != batch_size:
                    raise ValueError("step_value_forward_row_id must be unique within an actor forward.")
        if compute_step_value_probe and compute_similarity_step_embeddings:
            raise ValueError("Probe values and similarity embeddings cannot be requested in the same actor pass.")
        if compute_similarity_step_embeddings:
            if return_topk_logits:
                raise ValueError("Similarity step embeddings cannot be combined with return_topk_logits=True.")
            missing_keys = [key for key in ("step_start_mask", "step_end_mask") if key not in data.batch]
            if missing_keys:
                raise ValueError(
                    "compute_similarity_step_embeddings requires actor batch keys: " + ", ".join(missing_keys)
                )
            if "similarity_max_steps" not in data.meta_info:
                raise ValueError("compute_similarity_step_embeddings requires similarity_max_steps in meta_info.")
            similarity_max_steps = int(data.meta_info["similarity_max_steps"])
            if similarity_max_steps <= 0:
                raise ValueError("similarity_max_steps must be positive.")

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if compute_step_value_probe:
            select_keys.extend(["step_end_mask", "step_value_targets"])
            if delayed_audit_requested:
                select_keys.extend(audit_contract_keys)
        if compute_similarity_step_embeddings:
            select_keys.extend(["step_start_mask", "step_end_mask"])
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
        logits_lst = []
        logits_indices_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        step_values_lst = []
        step_value_trajectory_logit_mean_lst = []
        step_value_loss_lst = []
        step_value_hidden_lst = []
        step_value_target_lst = []
        step_value_endpoint_weight_lst = []
        step_value_audit_hidden_lst = []
        step_value_audit_sample_index_lst = []
        step_value_audit_mask_lst = []
        step_value_forward_row_id_lst = []
        similarity_step_embeddings_lst = []
        local_step_value_trajectory_count = 0
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            if compute_similarity_step_embeddings:
                step_start_mask = model_inputs["step_start_mask"].bool()
                similarity_step_end_mask = model_inputs["step_end_mask"].bool()
                if not torch.equal(step_start_mask.sum(dim=-1), similarity_step_end_mask.sum(dim=-1)):
                    raise ValueError("Similarity step start/end counts must match for every response.")
                model_inputs["step_end_mask"] = step_start_mask | similarity_step_end_mask
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    return_topk_logits=return_topk_logits,
                    return_step_hidden=compute_step_value_probe or compute_similarity_step_embeddings,
                )
            log_probs_lst.append(outputs["log_probs"])
            if compute_similarity_step_embeddings:
                boundary_hidden = outputs["step_hidden"].detach()
                boundary_mask = model_inputs["step_end_mask"].bool()
                boundary_ordinals = boundary_mask.reshape(-1).long().cumsum(dim=0) - 1
                start_indices = boundary_ordinals[step_start_mask.reshape(-1)]
                end_indices = boundary_ordinals[similarity_step_end_mask.reshape(-1)]
                if start_indices.numel() != end_indices.numel():
                    raise RuntimeError("Captured similarity step start/end counts differ.")
                step_embeddings = boundary_hidden.index_select(0, end_indices) - boundary_hidden.index_select(
                    0, start_indices
                )
                step_embeddings = torch.nn.functional.normalize(
                    step_embeddings.float(),
                    p=2,
                    dim=-1,
                    eps=1e-12,
                )
                step_counts = similarity_step_end_mask.sum(dim=-1, dtype=torch.long)
                packed_embeddings = torch.zeros(
                    (similarity_step_end_mask.shape[0], similarity_max_steps, step_embeddings.shape[-1]),
                    dtype=torch.float16,
                    device=step_embeddings.device,
                )
                offset = 0
                for row, count_tensor in enumerate(step_counts):
                    count = int(count_tensor.item())
                    packed_embeddings[row, :count] = step_embeddings[offset : offset + count].to(
                        dtype=packed_embeddings.dtype
                    )
                    offset += count
                if offset != step_embeddings.shape[0]:
                    raise RuntimeError("Failed to pack every similarity step embedding.")
                similarity_step_embeddings_lst.append(packed_embeddings)
            if compute_step_value_probe:
                step_value_hidden = outputs["step_hidden"].detach()
                self._ensure_step_value_probe_device(step_value_hidden.device)
                step_end_mask = model_inputs["step_end_mask"].bool()
                step_value_targets = model_inputs["step_value_targets"].float().reshape(-1)
                if step_value_targets.shape[0] != step_end_mask.shape[0]:
                    raise ValueError(
                        "step_value_targets must contain one scalar per trajectory, "
                        f"got shape {tuple(model_inputs['step_value_targets'].shape)}."
                    )
                if not torch.isfinite(step_value_targets).all():
                    raise ValueError("step_value_targets must be finite.")
                if ((step_value_targets < 0) | (step_value_targets > 1)).any():
                    raise ValueError("step_value_targets must be in [0, 1].")

                step_counts = step_end_mask.sum(dim=-1, dtype=torch.long)
                sample_indices = torch.arange(step_end_mask.shape[0], device=step_end_mask.device).repeat_interleave(
                    step_counts
                )
                endpoint_targets = step_value_targets.index_select(0, sample_indices)
                nonempty_counts = step_counts.clamp_min(1).float()
                endpoint_weights = nonempty_counts.reciprocal().index_select(0, sample_indices)

                with torch.no_grad():
                    endpoint_logits = self.step_value_probe(step_value_hidden.float()).squeeze(-1)
                    endpoint_values = torch.sigmoid(endpoint_logits)
                    endpoint_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                        endpoint_logits,
                        endpoint_targets,
                        reduction="none",
                    )
                trajectory_logit_means = torch.zeros(
                    step_end_mask.shape[0],
                    dtype=torch.float32,
                    device=step_end_mask.device,
                )
                trajectory_logit_means.scatter_add_(
                    0,
                    sample_indices,
                    endpoint_logits.float() * endpoint_weights,
                )
                dense_step_values = torch.zeros(
                    step_end_mask.shape,
                    dtype=torch.float32,
                    device=step_end_mask.device,
                )
                dense_step_values[step_end_mask] = endpoint_values
                endpoint_positions = torch.arange(
                    step_end_mask.shape[1],
                    device=step_end_mask.device,
                ).expand_as(step_end_mask)
                terminal_positions = endpoint_positions.masked_fill(~step_end_mask, -1).max(dim=-1).values
                if torch.any(terminal_positions < 0):
                    raise ValueError("Every probe trajectory must contain a terminal step endpoint.")
                dense_step_values[
                    torch.arange(step_end_mask.shape[0], device=step_end_mask.device),
                    terminal_positions,
                ] = step_value_targets
                trajectory_losses = torch.zeros(
                    step_end_mask.shape[0],
                    dtype=torch.float32,
                    device=step_end_mask.device,
                )
                trajectory_losses.scatter_add_(
                    0,
                    sample_indices,
                    endpoint_losses * endpoint_weights,
                )

                step_values_lst.append(dense_step_values)
                step_value_trajectory_logit_mean_lst.append(trajectory_logit_means)
                step_value_loss_lst.append(trajectory_losses)
                if delayed_audit_requested:
                    update_mask = model_inputs["step_value_probe_update_mask"].bool().reshape(-1)
                    audit_mask = model_inputs["step_value_prompt_center_audit_mask"].bool().reshape(-1)
                    endpoint_update_mask = update_mask.index_select(0, sample_indices)
                    endpoint_audit_mask = audit_mask.index_select(0, sample_indices)

                    step_value_hidden_lst.append(step_value_hidden[endpoint_update_mask])
                    step_value_target_lst.append(endpoint_targets[endpoint_update_mask])
                    step_value_endpoint_weight_lst.append(endpoint_weights[endpoint_update_mask])
                    local_step_value_trajectory_count += int(((step_counts > 0) & update_mask).sum().item())

                    # Audit hidden states stay local to the actor. After the one
                    # normal Probe update below, only the tiny MLP is replayed.
                    step_value_audit_hidden_lst.append(step_value_hidden[endpoint_audit_mask])
                    step_value_audit_sample_index_lst.append(sample_indices[endpoint_audit_mask])
                    step_value_audit_mask_lst.append(audit_mask)
                    step_value_forward_row_id_lst.append(model_inputs["step_value_forward_row_id"].reshape(-1))
                else:
                    # Preserve the legacy tensor and reduction path exactly when
                    # delayed auditing is disabled.
                    step_value_hidden_lst.append(step_value_hidden)
                    step_value_target_lst.append(endpoint_targets)
                    step_value_endpoint_weight_lst.append(endpoint_weights)
                    local_step_value_trajectory_count += int((step_counts > 0).sum().item())
            if return_topk_logits:
                logits_lst.append(outputs["logits"])
                logits_indices_lst.append(outputs["logits_indices"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if return_topk_logits:
            logits = torch.concat(logits_lst, dim=0)
            logits_indices = torch.concat(logits_indices_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)
        if compute_step_value_probe:
            step_values = torch.concat(step_values_lst, dim=0)
            step_value_trajectory_logit_mean = torch.concat(step_value_trajectory_logit_mean_lst, dim=0)
            step_value_probe_loss = torch.concat(step_value_loss_lst, dim=0)
        if compute_similarity_step_embeddings:
            similarity_step_embeddings = torch.concat(similarity_step_embeddings_lst, dim=0)
        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if return_topk_logits:
                logits = restore_dynamic_batch(logits, batch_idx_list)
                logits_indices = restore_dynamic_batch(logits_indices, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)
            if compute_step_value_probe:
                step_values = restore_dynamic_batch(step_values, batch_idx_list)
                step_value_trajectory_logit_mean = restore_dynamic_batch(
                    step_value_trajectory_logit_mean,
                    batch_idx_list,
                )
                step_value_probe_loss = restore_dynamic_batch(step_value_probe_loss, batch_idx_list)
            if compute_similarity_step_embeddings:
                similarity_step_embeddings = restore_dynamic_batch(similarity_step_embeddings, batch_idx_list)

        if compute_step_value_probe:
            all_step_value_hidden = torch.concat(step_value_hidden_lst, dim=0)
            all_step_value_targets = torch.concat(step_value_target_lst, dim=0)
            all_step_value_endpoint_weights = torch.concat(step_value_endpoint_weight_lst, dim=0)
            step_value_probe_grad_norm = self._update_step_value_probe(
                hidden=all_step_value_hidden,
                targets=all_step_value_targets,
                endpoint_weights=all_step_value_endpoint_weights,
                local_trajectory_count=local_step_value_trajectory_count,
                global_step=global_step,
            )
            if delayed_audit_requested:
                audit_ready_next = self._step_value_is_ready(global_step + 1)
                step_value_audit_trajectory_logit_mean_lst = []
                step_value_audit_endpoint_count_lst = []
                step_value_audit_ready_next_lst = []
                with torch.no_grad():
                    for audit_hidden, audit_sample_indices, audit_mask in zip(
                        step_value_audit_hidden_lst,
                        step_value_audit_sample_index_lst,
                        step_value_audit_mask_lst,
                        strict=True,
                    ):
                        row_count = audit_mask.shape[0]
                        audit_endpoint_counts = torch.zeros(
                            row_count,
                            dtype=torch.long,
                            device=audit_mask.device,
                        )
                        audit_endpoint_counts.scatter_add_(
                            0,
                            audit_sample_indices,
                            torch.ones_like(audit_sample_indices, dtype=torch.long),
                        )
                        audit_logit_sums = torch.zeros(
                            row_count,
                            dtype=torch.float32,
                            device=audit_mask.device,
                        )
                        if audit_hidden.numel() > 0:
                            audit_endpoint_logits = self.step_value_probe(audit_hidden.float()).squeeze(-1)
                            audit_logit_sums.scatter_add_(
                                0,
                                audit_sample_indices,
                                audit_endpoint_logits.float(),
                            )
                        audit_logit_means = audit_logit_sums / audit_endpoint_counts.clamp_min(1)
                        step_value_audit_trajectory_logit_mean_lst.append(audit_logit_means)
                        step_value_audit_endpoint_count_lst.append(audit_endpoint_counts)
                        step_value_audit_ready_next_lst.append(
                            audit_mask & (audit_endpoint_counts > 0) & audit_ready_next
                        )

                step_value_audit_trajectory_logit_mean = torch.concat(
                    step_value_audit_trajectory_logit_mean_lst,
                    dim=0,
                )
                step_value_audit_endpoint_count = torch.concat(step_value_audit_endpoint_count_lst, dim=0)
                step_value_audit_ready_next = torch.concat(step_value_audit_ready_next_lst, dim=0)
                step_value_forward_row_id = torch.concat(step_value_forward_row_id_lst, dim=0)
                if use_dynamic_bsz:
                    step_value_audit_trajectory_logit_mean = restore_dynamic_batch(
                        step_value_audit_trajectory_logit_mean,
                        batch_idx_list,
                    )
                    step_value_audit_endpoint_count = restore_dynamic_batch(
                        step_value_audit_endpoint_count,
                        batch_idx_list,
                    )
                    step_value_audit_ready_next = restore_dynamic_batch(
                        step_value_audit_ready_next,
                        batch_idx_list,
                    )
                    step_value_forward_row_id = restore_dynamic_batch(
                        step_value_forward_row_id,
                        batch_idx_list,
                    )

        outputs = {"log_probs": log_probs}
        if return_topk_logits:
            outputs["logits"] = logits
            outputs["logits_indices"] = logits_indices
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        if compute_step_value_probe:
            batch_size = log_probs.shape[0]
            outputs["step_values"] = step_values
            outputs["step_value_trajectory_logit_mean"] = step_value_trajectory_logit_mean
            outputs["step_value_ready"] = torch.full(
                (batch_size,),
                step_value_ready_before_update,
                dtype=torch.bool,
                device=log_probs.device,
            )
            outputs["step_value_probe_loss"] = step_value_probe_loss
            outputs["step_value_probe_grad_norm"] = (
                step_value_probe_grad_norm.to(device=log_probs.device).expand(batch_size).clone()
            )
            if delayed_audit_requested:
                outputs["step_value_audit_trajectory_logit_mean"] = step_value_audit_trajectory_logit_mean
                outputs["step_value_audit_endpoint_count"] = step_value_audit_endpoint_count
                outputs["step_value_audit_ready_next"] = step_value_audit_ready_next
                outputs["step_value_forward_row_id"] = step_value_forward_row_id
        if compute_similarity_step_embeddings:
            outputs["similarity_step_embeddings"] = similarity_step_embeddings
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

    @staticmethod
    def _shared_step_prefixes(
        response_mask: torch.Tensor,
        step_end_mask: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        """Map shared endpoint positions to response-prefix lengths."""
        valid_response_positions = response_mask.nonzero(as_tuple=False).flatten().tolist()
        selected_positions = step_end_mask.nonzero(as_tuple=False).flatten().tolist()
        response_position_to_prefix_length = {
            position: prefix_length for prefix_length, position in enumerate(valid_response_positions, start=1)
        }
        selected_prefix_lengths = [response_position_to_prefix_length[position] for position in selected_positions]
        return selected_positions, selected_prefix_lengths

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

    def _gather_answer_items(self, local_items: list[tuple[list[int], list[int], list[int], tuple[int, int, int]]]):
        process_group, rank, world_size = self._get_answer_dp_info()
        if world_size == 1:
            return local_items, rank, world_size

        device = torch.device(get_device_name(), get_device_id())
        metadata = []
        prompt_tokens = []
        answer_tokens = []
        answer_mask_tokens = []
        for prompt_ids, answer_ids, answer_mask, (owner_rank, sample_idx, answer_pos) in local_items:
            metadata.append([len(prompt_ids), len(answer_ids), owner_rank, sample_idx, answer_pos])
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
        max_payload_len = max_item_count * 5 + max_prompt_tokens + max_answer_tokens + max_answer_mask_tokens

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
            metadata_len = item_count * 5
            prompt_start = metadata_len
            answer_start = prompt_start + prompt_token_count
            answer_mask_start = answer_start + answer_token_count
            rank_payload = gathered_payloads[rank_idx]
            rank_metadata = rank_payload[:metadata_len].view(item_count, 5).cpu().tolist()
            rank_prompts = rank_payload[prompt_start:answer_start].cpu().tolist()
            rank_answers = rank_payload[answer_start : answer_start + answer_token_count].cpu().tolist()
            rank_answer_masks = (
                rank_payload[answer_mask_start : answer_mask_start + answer_mask_token_count].cpu().tolist()
            )

            prompt_offset = 0
            answer_offset = 0
            answer_mask_offset = 0
            for prompt_len, answer_len, owner_rank, sample_idx, answer_pos in rank_metadata:
                prompt_ids = rank_prompts[prompt_offset : prompt_offset + prompt_len]
                answer_ids = rank_answers[answer_offset : answer_offset + answer_len]
                answer_mask = rank_answer_masks[answer_mask_offset : answer_mask_offset + answer_len]
                prompt_offset += prompt_len
                answer_offset += answer_len
                answer_mask_offset += answer_len
                global_items.append((prompt_ids, answer_ids, answer_mask, (owner_rank, sample_idx, answer_pos)))
        return global_items, rank, world_size

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_answer_log_prob(self, data: DataProto, tokenizer) -> dict[str, torch.Tensor]:
        """Compute old-policy answer log probabilities at shared step endpoints.

        Step boundaries are supplied by ``verl.utils.step_split`` through the
        input ``step_end_mask``. This method only scores the selected prefixes;
        it does not perform any independent response splitting.

        Returns:
            dict[str, torch.Tensor]:
                log_probs: (batch_size, response_length + 1)
        """
        self.actor_module.eval()

        # if "multi_modal_inputs" in data.non_tensor_batch.keys():
        #     raise NotImplementedError("compute_answer_log_prob currently supports text-only batches.")

        micro_batch_size = max(int(data.meta_info["micro_batch_size"]), 1)
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        answer_prefix = self._decode_common_escapes(data.meta_info["answer_prefix"])
        data = data.select(
            batch_keys=["responses", "input_ids", "attention_mask", "response_mask", "step_end_mask"],
            non_tensor_batch_keys=["reward_model"],
        )

        responses = data.batch["responses"]
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        response_mask = data.batch["response_mask"]
        step_end_mask = data.batch["step_end_mask"].to(device=response_mask.device, dtype=torch.bool)
        response_len = responses.size(1)
        prompt_len = input_ids.size(1) - response_len
        batch_size = responses.size(0)
        device = torch.device(get_device_name(), get_device_id())
        dtype = responses.dtype

        if step_end_mask.shape != response_mask.shape:
            raise ValueError(
                "step_end_mask must match response_mask in answer-log-prob computation, "
                f"got {tuple(step_end_mask.shape)} and {tuple(response_mask.shape)}."
            )
        response_mask_bool = response_mask.bool()
        if torch.any(step_end_mask & ~response_mask_bool):
            raise ValueError("step_end_mask may select only valid response tokens.")
        if torch.any(step_end_mask.sum(dim=-1) < 1):
            raise ValueError("Shared step_end_mask must contain at least one endpoint per response.")

        step_end_mask = step_end_mask.to(device=device)
        answer_log_probs = torch.zeros((batch_size, response_len + 1), dtype=torch.float32, device=device)
        computed_pos_mask = torch.zeros((batch_size, response_len + 1), dtype=torch.bool, device=device)
        computed_pos_mask[:, 0] = True
        computed_pos_mask[:, 1:] = step_end_mask

        _, rank, world_size = self._get_answer_dp_info()
        sample_records = []
        ground_truth_encoding_cache: dict[str, tuple[list[int], list[int]]] = {}

        for i in range(batch_size):
            prompt_token_ids = input_ids[i, :prompt_len][attention_mask[i, :prompt_len].bool()].tolist()
            valid_response_positions = response_mask[i].nonzero(as_tuple=False).flatten().tolist()
            valid_response_ids = responses[i, valid_response_positions].tolist()
            selected_positions, selected_prefix_lengths = self._shared_step_prefixes(
                response_mask[i],
                step_end_mask[i],
            )

            reward_model_data = data.non_tensor_batch["reward_model"][i]
            ground_truth = self._extract_ground_truth(reward_model_data)
            if ground_truth not in ground_truth_encoding_cache:
                ground_truth_encoding_cache[ground_truth] = self._encode_prefixed_answer(
                    tokenizer,
                    answer_prefix,
                    ground_truth,
                )
            encoded_answer, encoded_answer_mask = ground_truth_encoding_cache[ground_truth]

            sample_records.append(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "valid_response_ids": valid_response_ids,
                    "selected_positions": selected_positions,
                    "selected_prefix_lengths": selected_prefix_lengths,
                    "encoded_answer": encoded_answer,
                    "encoded_answer_mask": encoded_answer_mask,
                }
            )

        local_items: list[tuple[list[int], list[int], list[int], tuple[int, int, int]]] = []
        for i, sample_record in enumerate(sample_records):
            prompt_token_ids = sample_record["prompt_token_ids"]
            valid_response_ids = sample_record["valid_response_ids"]
            selected_positions = sample_record["selected_positions"]
            selected_prefix_lengths = sample_record["selected_prefix_lengths"]
            encoded_answer = sample_record["encoded_answer"]
            encoded_answer_mask = sample_record["encoded_answer_mask"]
            local_items.append((prompt_token_ids, encoded_answer, encoded_answer_mask, (rank, i, 0)))

            for pos, prefix_len in zip(selected_positions, selected_prefix_lengths, strict=True):
                prefix_token_ids = prompt_token_ids + valid_response_ids[:prefix_len]
                local_items.append((prefix_token_ids, encoded_answer, encoded_answer_mask, (rank, i, pos + 1)))

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
        for global_item_idx, (_, _, _, (owner_rank, sample_idx, answer_pos)) in enumerate(global_items):
            if owner_rank == rank:
                owned_item_indices.append(global_item_idx)
                owned_sample_indices.append(sample_idx)
                owned_answer_positions.append(answer_pos)

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

        answer_pos_ids = torch.arange(response_len + 1, dtype=torch.long, device=device).unsqueeze(0)
        last_computed_pos = torch.where(computed_pos_mask, answer_pos_ids, torch.zeros_like(answer_pos_ids))
        last_computed_pos = last_computed_pos.cummax(dim=1).values
        answer_log_probs = answer_log_probs.gather(dim=1, index=last_computed_pos)

        return {"log_probs": answer_log_probs}

    @torch.no_grad()
    def _precompute_dapo_reference_old_policy_topk(
        self,
        data: DataProto,
        temperature: float,
        top_k: int,
        pad_token_id: int,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Cache privileged-context targets before any actor optimizer step."""

        loss_mask = data.batch["dapo_reference_kl_loss_mask"].bool()
        selected_rows = loss_mask.any(dim=-1)
        selected_indices = selected_rows.nonzero(as_tuple=False).flatten()
        teacher_micro_batches = []
        if selected_indices.numel() > 0:
            teacher_loss_mask = data.batch["dapo_reference_kl_loss_mask"][selected_indices]
            teacher_data = DataProto.from_dict(
                tensors={
                    "responses": data.batch["responses"][selected_indices],
                    "response_mask": teacher_loss_mask,
                    "input_ids": data.batch["dapo_reference_teacher_input_ids"][selected_indices],
                    "attention_mask": data.batch["dapo_reference_teacher_attention_mask"][selected_indices],
                    "position_ids": data.batch["dapo_reference_teacher_position_ids"][selected_indices],
                    "dapo_reference_kl_loss_mask": teacher_loss_mask,
                    "dapo_reference_kl_row_id": data.batch["dapo_reference_kl_row_id"][selected_indices],
                },
                meta_info=data.meta_info,
            )

            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                teacher_micro_batches, _ = prepare_dynamic_batch(
                    teacher_data,
                    max_token_len=max_token_len,
                    dp_group=None,
                    same_micro_num_in_dp=False,
                )
            else:
                teacher_micro_batches = teacher_data.split(self.config.ppo_micro_batch_size_per_gpu)

        local_micro_batch_count = len(teacher_micro_batches)
        sync_group = self._get_micro_batch_sync_group()
        sync_device = "cpu"
        if torch.distributed.is_initialized():
            sync_backend = str(torch.distributed.get_backend(group=sync_group)).lower()
            if sync_backend in {"nccl", "hccl"}:
                sync_device = get_device_name()
        synchronized_micro_batch_count = torch.tensor(
            local_micro_batch_count,
            dtype=torch.long,
            device=sync_device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                synchronized_micro_batch_count,
                op=torch.distributed.ReduceOp.MAX,
                group=sync_group,
            )
        synchronized_micro_batch_count = int(synchronized_micro_batch_count.item())
        if synchronized_micro_batch_count == 0:
            return {}

        dummy_loss_mask = data.batch["response_mask"][:1]
        dummy_teacher_data = DataProto.from_dict(
            tensors={
                "responses": data.batch["responses"][:1],
                "response_mask": dummy_loss_mask,
                "input_ids": data.batch["dapo_reference_teacher_input_ids"][:1],
                "attention_mask": data.batch["dapo_reference_teacher_attention_mask"][:1],
                "position_ids": data.batch["dapo_reference_teacher_position_ids"][:1],
                "dapo_reference_kl_loss_mask": dummy_loss_mask,
                "dapo_reference_kl_row_id": torch.full((1,), -1, dtype=torch.long),
            },
            meta_info=data.meta_info,
        )

        teacher_cache = {}
        was_training = self.actor_module.training
        try:
            self.actor_module.eval()
            for micro_batch_index in range(synchronized_micro_batch_count):
                is_dummy_micro_batch = micro_batch_index >= local_micro_batch_count
                teacher_micro_batch = (
                    dummy_teacher_data if is_dummy_micro_batch else teacher_micro_batches[micro_batch_index]
                )
                teacher_micro_batch = teacher_micro_batch.to(get_device_id())
                teacher_inputs = {
                    **teacher_micro_batch.batch,
                    **teacher_micro_batch.non_tensor_batch,
                    "pad_token_id": pad_token_id,
                }
                teacher_logits = self._forward_micro_batch(
                    teacher_inputs,
                    temperature=temperature,
                    return_valid_logits=True,
                    valid_logits_only=True,
                )["valid_logits"]
                if is_dummy_micro_batch:
                    teacher_micro_batch.to("cpu")
                    del teacher_inputs, teacher_logits
                    continue
                top_indices, top_log_probs, tail_probs = summarize_dapo_reference_teacher_topk(
                    teacher_logits,
                    top_k,
                )

                row_ids = teacher_inputs["dapo_reference_kl_row_id"].detach().cpu().tolist()
                token_counts = teacher_inputs["dapo_reference_kl_loss_mask"].sum(dim=-1).detach().cpu().tolist()
                offset = 0
                for row_id, token_count in zip(row_ids, token_counts, strict=True):
                    row_id = int(row_id)
                    token_count = int(token_count)
                    end = offset + token_count
                    if row_id in teacher_cache:
                        raise ValueError(f"DAPO reference KL old-policy cache contains duplicate row {row_id}.")
                    teacher_cache[row_id] = (
                        top_indices[offset:end].to(device="cpu", dtype=torch.int32).contiguous(),
                        top_log_probs[offset:end].to(device="cpu").contiguous(),
                        tail_probs[offset:end].to(device="cpu").contiguous(),
                    )
                    offset = end
                if offset != teacher_logits.shape[0]:
                    raise ValueError("DAPO reference KL old-policy cache token counts do not match the teacher output.")
                teacher_micro_batch.to("cpu")
                del teacher_inputs, teacher_logits, top_indices, top_log_probs, tail_probs
        finally:
            self._reshard_model_after_forward(self.actor_module)
            self.actor_module.train(was_training)

        if len(teacher_cache) != int(selected_rows.sum().item()):
            raise ValueError("DAPO reference KL old-policy cache is missing selected response rows.")
        return teacher_cache

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        reference_kl_enabled = bool(data.meta_info.get("dapo_reference_kl", False))
        reference_kl_coef = float(data.meta_info.get("dapo_reference_kl_coef", 0.0))
        reference_kl_temperature = float(data.meta_info.get("dapo_reference_kl_temperature", 1.0))
        reference_kl_approximation = str(data.meta_info.get("dapo_reference_kl_approximation", "topk"))
        reference_kl_top_k = int(data.meta_info.get("dapo_reference_kl_top_k", 100))
        reference_kl_token_chunk_size = int(
            data.meta_info.get("dapo_reference_kl_token_chunk_size", _DAPO_REFERENCE_KL_TOKEN_CHUNK_SIZE)
        )

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if reference_kl_enabled:
            reference_kl_keys = [
                "dapo_reference_teacher_input_ids",
                "dapo_reference_teacher_attention_mask",
                "dapo_reference_teacher_position_ids",
                "dapo_reference_kl_loss_mask",
            ]
            missing_reference_kl_keys = [key for key in reference_kl_keys if key not in data.batch]
            if missing_reference_kl_keys:
                raise ValueError(f"DAPO reference KL actor update requires batch keys {missing_reference_kl_keys}.")
            select_keys.extend(reference_kl_keys)
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        kl_loss_needs_ref = self.config.use_kl_loss and float(self.config.kl_loss_coef) != 0.0
        if kl_loss_needs_ref or loss_mode in {"ours", "my_future"}:
            select_keys.append("ref_log_prob")
        if loss_mode == "dgpo":
            select_keys.extend(["ref_logits", "ref_logits_indices"])
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

        reference_teacher_cache = {}
        if reference_kl_enabled and reference_kl_coef != 0.0:
            if reference_kl_approximation != "topk":
                raise ValueError(
                    "Old-policy DAPO reference KL requires approximation='topk' because full-vocabulary "
                    "teacher logits cannot be cached for the complete prompt batch."
                )
            data.batch["dapo_reference_kl_row_id"] = torch.arange(len(data), dtype=torch.long)
            reference_teacher_cache = self._precompute_dapo_reference_old_policy_topk(
                data=data,
                temperature=reference_kl_temperature,
                top_k=reference_kl_top_k,
                pad_token_id=pad_token_id,
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
            "actor/dapo_reference_kl_loss": 0.0,
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
                        self.config.calculate_entropy or (entropy_coeff != 0) or loss_mode in {"ours", "dgpo", "my"}
                    )

                    reference_kl_mask = model_inputs.get("dapo_reference_kl_loss_mask")
                    has_reference_kl = bool(
                        reference_kl_enabled
                        and reference_kl_coef != 0.0
                        and reference_kl_mask is not None
                        and reference_kl_mask.any().item()
                    )
                    teacher_top_indices = None
                    teacher_top_log_probs = None
                    teacher_tail_prob = None
                    if has_reference_kl:
                        teacher_top_indices, teacher_top_log_probs, teacher_tail_prob = (
                            gather_dapo_reference_teacher_topk(
                                teacher_cache=reference_teacher_cache,
                                row_ids=model_inputs["dapo_reference_kl_row_id"],
                                loss_mask=reference_kl_mask,
                                device=reference_kl_mask.device,
                            )
                        )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_topk_logits=loss_mode == "dgpo",
                        return_valid_logits=has_reference_kl,
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
                    elif loss_mode == "my":
                        pg_loss, pg_metrics, _ = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            entropy=entropy,
                            vinfo_weights=model_inputs.get("vinfo_weights"),
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                    elif loss_mode == "my_future":
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

                    reference_kl_loss = policy_loss.new_zeros(())
                    if has_reference_kl:
                        student_logits = outputs["valid_logits"]
                        if reference_kl_temperature != temperature:
                            student_logits = student_logits * (temperature / reference_kl_temperature)
                        valid_token_reference_kl = dapo_reference_topk_forward_kl(
                            student_logits=student_logits,
                            teacher_top_indices=teacher_top_indices,
                            teacher_top_log_probs=teacher_top_log_probs,
                            teacher_tail_prob=teacher_tail_prob,
                            token_chunk_size=reference_kl_token_chunk_size,
                        )
                        token_reference_kl = torch.zeros_like(
                            reference_kl_mask, dtype=valid_token_reference_kl.dtype
                        ).masked_scatter(reference_kl_mask.bool(), valid_token_reference_kl)
                        reference_kl_loss = agg_loss(
                            loss_mat=token_reference_kl,
                            loss_mask=reference_kl_mask,
                            loss_agg_mode="token-mean",
                        )
                        metrics["actor/dapo_reference_kl_loss"] += reference_kl_loss.detach().item() * loss_scale_factor

                    if reference_kl_enabled and reference_kl_coef != 0.0:
                        policy_loss = mix_dapo_reference_kl_loss(
                            dapo_loss=policy_loss,
                            reference_kl_loss=reference_kl_loss,
                            loss_coef=reference_kl_coef,
                        )
                        # This metric must be emitted for every synchronized micro-batch on every rank.
                        # Omitting all-correct micro-batches creates ragged per-rank metric lists that
                        # DataProto.concat cannot reduce with numpy.
                        micro_batch_metrics["actor/dapo_reference_kl_tokens"] = int(reference_kl_mask.sum().item())
                        micro_batch_metrics["actor/dapo_reference_kl_coef"] = reference_kl_coef

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
        return metrics
