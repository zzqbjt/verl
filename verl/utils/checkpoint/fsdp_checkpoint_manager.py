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

import json
import logging
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, exists, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import (
    fsdp_version,
    get_fsdp_checkpoint_shard_info,
    get_fsdp_full_state_dict,
    get_fsdp_state_ctx,
)
from verl.utils.logger import log_with_rank
from verl.utils.transformers_compat import get_auto_model_for_vision2seq

from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
        shard_world_size (int): Number of unique FSDP model/optimizer shards.
        format_version (int): Checkpoint layout version.
    """

    FSDP_version: int
    world_size: int
    shard_world_size: int
    format_version: int = 2


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads one model & optimizer file per unique FSDP shard
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
        trust_remote_code: Whether to trust_remote_code when loading the model configuration
        device_mesh: Optional FSDP device mesh used to remove HSDP/DDP replica duplication.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        trust_remote_code: bool = False,
        device_mesh=None,
        **kwargs,
    ):
        if processing_class is None and "tokenizer" in kwargs:
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )
        self.trust_remote_code = trust_remote_code
        self.device_mesh = device_mesh if device_mesh is not None else self._infer_fsdp2_device_mesh()
        self.shard_world_size, self.shard_rank, self.shard_writer_rank = self._get_checkpoint_shard_info()

    def _infer_fsdp2_device_mesh(self):
        """Best-effort mesh discovery for callers that predate ``device_mesh``."""
        if fsdp_version(self.model) != 2:
            return None
        for tensor in self.model.parameters():
            device_mesh = getattr(tensor, "device_mesh", None)
            if device_mesh is not None:
                return device_mesh
        return None

    def _get_checkpoint_shard_info(self):
        """Map this process to a unique FSDP2 model/optimizer checkpoint shard."""
        # FSDP1 sharded state dicts contain rank-specific ShardedTensor
        # placement metadata, so keep their historical per-global-rank layout.
        if fsdp_version(self.model) != 2 or self.device_mesh is None:
            return self.world_size, self.rank, self.rank
        return get_fsdp_checkpoint_shard_info(self.device_mesh, self.rank)

    def _resolve_sharded_checkpoint_path(self, local_path: str, prefix: str) -> str:
        """Prefer the deduplicated v2 shard name, then accept the legacy name."""
        candidates = [
            os.path.join(
                local_path,
                f"{prefix}_world_size_{self.shard_world_size}_rank_{self.shard_rank}.pt",
            ),
            os.path.join(
                local_path,
                f"{prefix}_world_size_{self.world_size}_rank_{self.rank}.pt",
            ),
        ]
        candidates = list(dict.fromkeys(candidates))
        path = next((candidate for candidate in candidates if exists(candidate)), None)
        if path is None:
            raise FileNotFoundError(
                f"Could not find {prefix} checkpoint shard; tried: {candidates}"
            )
        return path

    def _model_buffer_checkpoint_path(self, local_path: str) -> str:
        return os.path.join(
            local_path,
            f"model_buffers_world_size_{self.world_size}_rank_{self.rank}.pt",
        )

    def _persistent_buffer_keys(self, state_dict: dict) -> set[str]:
        """Return model buffer names that are present in its persistent state dict."""
        buffer_names = {name for name, _ in self.model.named_buffers()}
        return buffer_names & set(state_dict)

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # check if the checkpoint_load_contents is valid
        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )

        # Replica ranks may load the same unique FSDP shard in format v2.
        local_paths_to_delete = []
        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_model
            else None
        )
        optim_cfg = (
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_optimizer
            else None
        )
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if self.should_load_model:
                remote_model_path = self._resolve_sharded_checkpoint_path(local_path, "model")
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                new_model_path = os.path.join(
                    local_path,
                    f"model_world_size_{self.shard_world_size}_rank_{self.shard_rank}.pt",
                )
                uses_deduplicated_model = (
                    self.shard_world_size < self.world_size
                    and remote_model_path == new_model_path
                )
                if uses_deduplicated_model:
                    current_state_dict = self.model.state_dict()
                    persistent_buffer_keys = self._persistent_buffer_keys(current_state_dict)
                    remote_buffer_path = self._model_buffer_checkpoint_path(local_path)
                    if persistent_buffer_keys:
                        if not exists(remote_buffer_path):
                            raise FileNotFoundError(
                                "Deduplicated model checkpoint is missing rank-local buffer state: "
                                f"{remote_buffer_path}"
                            )
                        local_buffer_path = copy_to_local(remote_buffer_path)
                        buffer_state_dict = torch.load(local_buffer_path, weights_only=False)
                        if set(buffer_state_dict) != persistent_buffer_keys:
                            raise ValueError(
                                "Rank-local model buffer keys do not match the model: "
                                f"saved={sorted(buffer_state_dict)}, expected={sorted(persistent_buffer_keys)}"
                            )
                        model_state_dict.update(buffer_state_dict)
                        if del_local_after_load and is_non_local(remote_buffer_path):
                            local_paths_to_delete.append(local_buffer_path)
                self.model.load_state_dict(model_state_dict)
                log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)
                if del_local_after_load and is_non_local(remote_model_path):
                    local_paths_to_delete.append(local_model_path)

            if self.should_load_optimizer:
                remote_optim_path = self._resolve_sharded_checkpoint_path(local_path, "optim")
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                self.optimizer.load_state_dict(optimizer_state_dict)
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)
                if del_local_after_load and is_non_local(remote_optim_path):
                    local_paths_to_delete.append(local_optim_path)

        if self.should_load_extra:
            remote_extra_state_path = os.path.join(
                local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            if del_local_after_load and is_non_local(remote_extra_state_path):
                local_paths_to_delete.append(local_extra_state_path)
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        # wait for everyone to load checkpoints
        torch.distributed.barrier()
        if del_local_after_load:
            # Multiple replica ranks on one node can share a cached shard path.
            # They may remove it only after every rank has completed torch.load.
            for path in dict.fromkeys(local_paths_to_delete):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    log_with_rank(
                        f"remove local resume ckpt file failed, exception {e} will be ignored",
                        rank=self.rank,
                        logger=logger,
                    )
            torch.distributed.barrier()

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        if self.rank == 0:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )

        # Save model/optimizer only once per unique FSDP shard. Extra state is
        # still per global rank because RNG state must not be deduplicated.
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(
                    local_path,
                    f"model_world_size_{self.shard_world_size}_rank_{self.shard_rank}.pt",
                )
                optim_path = os.path.join(
                    local_path,
                    f"optim_world_size_{self.shard_world_size}_rank_{self.shard_rank}.pt",
                )
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.should_save_model:
                    has_buffers = any(True for _ in self.model.named_buffers())
                    needs_buffer_state = self.shard_world_size < self.world_size and has_buffers
                    model_state_dict = None
                    if self.rank == self.shard_writer_rank or needs_buffer_state:
                        model_state_dict = self.model.state_dict()

                    if self.rank == self.shard_writer_rank:
                        torch.save(model_state_dict, model_path)
                        log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                    if needs_buffer_state:
                        persistent_buffer_keys = self._persistent_buffer_keys(model_state_dict)
                        if persistent_buffer_keys:
                            buffer_state_dict = {
                                key: model_state_dict[key]
                                for key in persistent_buffer_keys
                            }
                            buffer_path = self._model_buffer_checkpoint_path(local_path)
                            torch.save(buffer_state_dict, buffer_path)
                            log_with_rank(
                                f"Saved rank-local model buffers to {os.path.abspath(buffer_path)}",
                                rank=self.rank,
                                logger=logger,
                            )

                if self.should_save_optimizer and self.rank == self.shard_writer_rank:
                    optimizer_state_dict = self.optimizer.state_dict()
                    torch.save(optimizer_state_dict, optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        if self.rank == 0:
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            generation_config = None
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:
                    # if the generation config isn't available, we don't save it
                    pass

            if hasattr(model_config, "auto_map") and None in model_config.auto_map:
                model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}

            model_config.save_pretrained(hf_config_tokenizer_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
            # loaded from the Hub.
            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
                shard_world_size=self.shard_world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.should_save_hf_model:
            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    auto_model_cls = get_auto_model_for_vision2seq()
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(
                        model_config, torch_dtype=torch.bfloat16, trust_remote_code=self.trust_remote_code
                    )

                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found "
                            f"in, using a generation config created from the model config when saving hf_model."
                        )

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        if self.rank == 0:
            self.register_checkpoint(local_path, max_ckpt_to_keep)
