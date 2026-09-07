# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
from verl.workers import fsdp_workers


@pytest.fixture
def init_cpu_worker(monkeypatch):
    """Exercise the real constructor without processes, device meshes, or models."""
    monkeypatch.setattr(fsdp_workers.Worker, "__init__", lambda self: setattr(self, "_rank", 0))
    monkeypatch.setattr(fsdp_workers.Worker, "_register_dispatch_collect_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(fsdp_workers.DistProfilerExtension, "__init__", lambda *args, **kwargs: None)
    monkeypatch.setattr(fsdp_workers, "DistProfiler", MagicMock())
    monkeypatch.setattr(fsdp_workers, "FSDPUlyssesShardingManager", MagicMock())
    monkeypatch.setattr(fsdp_workers.torch.distributed, "is_initialized", lambda: True)
    init_process_group = MagicMock(side_effect=AssertionError("CPU test must not create a process group"))
    monkeypatch.setattr(fsdp_workers.torch.distributed, "init_process_group", init_process_group)
    monkeypatch.setattr(fsdp_workers.torch.cuda, "init", MagicMock(side_effect=AssertionError("No CUDA allowed")))

    def initialize(config, world_size, role="actor_rollout_ref"):
        mesh = MagicMock()
        mesh.size.return_value = world_size
        with (
            patch.object(fsdp_workers.torch.distributed, "get_world_size", return_value=world_size),
            patch.object(fsdp_workers, "create_device_mesh", return_value=mesh),
            patch("verl.workers.engine.fsdp.utils.apply_npu_fsdp_patches"),
        ):
            worker = fsdp_workers.ActorRolloutRefWorker(config, role)
        init_process_group.assert_not_called()
        return worker

    return initialize


def worker_config(base_n, prompt_mini_batch_size, training_n=None):
    config = OmegaConf.create(
        {
            "model": {},
            "actor": {
                "fsdp_config": {"fsdp_size": -1},
                "ppo_mini_batch_size": prompt_mini_batch_size,
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": None,
            },
            "rollout": {"n": base_n, "log_prob_micro_batch_size": None},
            "ref": {"fsdp_config": {}, "log_prob_micro_batch_size": None},
        }
    )
    if training_n is not None:
        config.actor.rollout_n = training_n
    return config


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("prompt_mini_batch_size", [8, 16])
@pytest.mark.parametrize(
    ("base_n", "training_n"),
    [
        (8, None),  # Legacy configurations without actor.rollout_n.
        (8, 8),  # Credit-only / ordinary DAPO-8.
        (16, 16),  # Ordinary DAPO-16.
        (8, 16),  # Full method, Sparse-Only, and lambda=0 with 2x4 branches.
        (6, 12),  # Formula is not a hard-coded response count.
    ],
)
def test_fsdp_actor_normalizes_with_training_response_count(
    init_cpu_worker, world_size, prompt_mini_batch_size, base_n, training_n
):
    config = worker_config(base_n, prompt_mini_batch_size, training_n)
    worker = init_cpu_worker(config, world_size)
    effective_n = training_n if training_n is not None else base_n
    local_mini_batch_size = prompt_mini_batch_size * effective_n // world_size
    assert worker.config.actor.ppo_mini_batch_size == local_mini_batch_size
    assert worker.config.rollout.n == base_n
    local_training_batch_size = 128 * effective_n // world_size
    assert local_training_batch_size // local_mini_batch_size == 128 // prompt_mini_batch_size


@pytest.mark.parametrize("role", ["rollout", "ref"])
def test_non_actor_workers_do_not_normalize_actor_minibatch(init_cpu_worker, role):
    config = worker_config(base_n=8, prompt_mini_batch_size=8, training_n=16)
    init_cpu_worker(config, world_size=4, role=role)
    assert config.actor.ppo_mini_batch_size == 8
    assert config.rollout.n == 8


@pytest.mark.parametrize("train_mc_branches", [False, True])
@pytest.mark.parametrize("prompt_mini_batch_size", [8, 16])
def test_dapo_training_count_reaches_real_fsdp_worker(init_cpu_worker, train_mc_branches, prompt_mini_batch_size):
    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": worker_config(8, prompt_mini_batch_size, training_n=8),
            "algorithm": {
                "sparse_counterfactual_credit": {
                    "enabled": True,
                    "train_mc_branches": train_mc_branches,
                    "branch_groups_per_prompt": 2,
                    "num_samples": 4,
                }
            },
        }
    )
    OmegaConf.set_struct(trainer.config, True)
    effective_n = 16 if train_mc_branches else 8
    workers = []

    def initialize_worker():
        # Ray serializes this config; the worker's per-rank normalization must
        # not overwrite the prompt-based mini-batch in the trainer's config.
        workers.append(init_cpu_worker(deepcopy(trainer.config.actor_rollout_ref), world_size=4))

    with patch("recipe.dapo.dapo_ray_trainer.RayPPOTrainer.init_workers", side_effect=initialize_worker):
        trainer.init_workers()

    assert trainer.config.actor_rollout_ref.rollout.n == 8
    assert trainer.config.actor_rollout_ref.actor.rollout_n == effective_n
    assert trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size == prompt_mini_batch_size
    assert workers[0].config.actor.ppo_mini_batch_size == prompt_mini_batch_size * effective_n // 4
