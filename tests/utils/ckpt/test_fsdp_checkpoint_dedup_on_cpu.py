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

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


class _FakeModel:
    def __init__(self, rank):
        self.state_dict_calls = 0
        self.loaded_state = None
        self.running_value = torch.tensor([float(rank)])

    def parameters(self):
        return iter(())

    def state_dict(self):
        self.state_dict_calls += 1
        return {
            "weight": torch.tensor([11.0]),
            "running_value": self.running_value.clone(),
        }

    def named_buffers(self):
        return iter((("running_value", self.running_value),))

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


class _FakeOptimizer:
    def __init__(self):
        self.state_dict_calls = 0
        self.loaded_state = None

    def state_dict(self):
        self.state_dict_calls += 1
        return {"state": {0: {"step": torch.tensor(3)}}}

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


def _device_mesh():
    return SimpleNamespace(
        mesh=torch.tensor([[0, 1], [2, 3]]),
        mesh_dim_names=("ddp", "fsdp"),
    )


def _make_manager(monkeypatch, rank, checkpoint_config, fsdp_version=2):
    import verl.utils.checkpoint.fsdp_checkpoint_manager as checkpoint_module

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(checkpoint_module, "fsdp_version", lambda model: fsdp_version)
    monkeypatch.setattr(checkpoint_module, "get_fsdp_state_ctx", lambda *args, **kwargs: nullcontext())

    model = _FakeModel(rank)
    optimizer = _FakeOptimizer()
    manager = checkpoint_module.FSDPCheckpointManager(
        model=model,
        optimizer=optimizer,
        checkpoint_config=checkpoint_config,
        device_mesh=_device_mesh(),
    )
    return manager, model, optimizer


def test_save_writes_model_and_optimizer_only_for_unique_fsdp_shard(monkeypatch, tmp_path):
    checkpoint_config = {
        "load_contents": [],
        "save_contents": ["model", "optimizer", "extra"],
    }
    writer, writer_model, writer_optimizer = _make_manager(monkeypatch, 1, checkpoint_config)
    writer.save_checkpoint(str(tmp_path))

    replica, replica_model, replica_optimizer = _make_manager(monkeypatch, 3, checkpoint_config)
    replica.save_checkpoint(str(tmp_path))

    assert (tmp_path / "model_world_size_2_rank_1.pt").is_file()
    assert (tmp_path / "optim_world_size_2_rank_1.pt").is_file()
    assert not (tmp_path / "model_world_size_4_rank_3.pt").exists()
    assert not (tmp_path / "optim_world_size_4_rank_3.pt").exists()
    assert (tmp_path / "extra_state_world_size_4_rank_1.pt").is_file()
    assert (tmp_path / "extra_state_world_size_4_rank_3.pt").is_file()
    assert (tmp_path / "model_buffers_world_size_4_rank_1.pt").is_file()
    assert (tmp_path / "model_buffers_world_size_4_rank_3.pt").is_file()
    assert writer_model.state_dict_calls == 1
    assert writer_optimizer.state_dict_calls == 1
    assert replica_model.state_dict_calls == 1
    assert replica_optimizer.state_dict_calls == 0


def test_load_maps_replica_to_new_unique_shard(monkeypatch, tmp_path):
    torch.save(
        {"weight": torch.tensor([17.0]), "running_value": torch.tensor([1.0])},
        tmp_path / "model_world_size_2_rank_1.pt",
    )
    torch.save(
        {"running_value": torch.tensor([3.0])},
        tmp_path / "model_buffers_world_size_4_rank_3.pt",
    )
    torch.save({"state": {0: {"step": torch.tensor(9)}}}, tmp_path / "optim_world_size_2_rank_1.pt")
    checkpoint_config = {
        "load_contents": ["model", "optimizer"],
        "save_contents": [],
    }
    manager, model, optimizer = _make_manager(monkeypatch, 3, checkpoint_config)

    manager.load_checkpoint(str(tmp_path))

    torch.testing.assert_close(model.loaded_state["weight"], torch.tensor([17.0]))
    torch.testing.assert_close(model.loaded_state["running_value"], torch.tensor([3.0]))
    assert optimizer.loaded_state["state"][0]["step"].item() == 9


def test_load_rejects_deduplicated_model_without_rank_local_buffers(monkeypatch, tmp_path):
    torch.save(
        {"weight": torch.tensor([17.0]), "running_value": torch.tensor([1.0])},
        tmp_path / "model_world_size_2_rank_1.pt",
    )
    checkpoint_config = {
        "load_contents": ["model"],
        "save_contents": [],
    }
    manager, _, _ = _make_manager(monkeypatch, 3, checkpoint_config)

    with pytest.raises(FileNotFoundError, match="rank-local buffer state"):
        manager.load_checkpoint(str(tmp_path))


def test_load_falls_back_to_legacy_global_rank_files(monkeypatch, tmp_path):
    torch.save(
        {"weight": torch.tensor([23.0]), "running_value": torch.tensor([3.0])},
        tmp_path / "model_world_size_4_rank_3.pt",
    )
    torch.save({"state": {0: {"step": torch.tensor(12)}}}, tmp_path / "optim_world_size_4_rank_3.pt")
    checkpoint_config = {
        "load_contents": ["model", "optimizer"],
        "save_contents": [],
    }
    manager, model, optimizer = _make_manager(monkeypatch, 3, checkpoint_config)

    manager.load_checkpoint(str(tmp_path))

    torch.testing.assert_close(model.loaded_state["weight"], torch.tensor([23.0]))
    assert optimizer.loaded_state["state"][0]["step"].item() == 12


def test_fsdp1_keeps_per_global_rank_checkpoint_layout(monkeypatch):
    checkpoint_config = {"load_contents": [], "save_contents": []}
    manager, _, _ = _make_manager(monkeypatch, 3, checkpoint_config, fsdp_version=1)

    assert (manager.shard_world_size, manager.shard_rank, manager.shard_writer_rank) == (4, 3, 3)


def test_model_merger_prefers_unique_shard_world_size(tmp_path):
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    with open(tmp_path / "fsdp_config.json", "w") as config_file:
        json.dump({"world_size": 4, "shard_world_size": 1, "format_version": 2}, config_file)
    merger = object.__new__(FSDPModelMerger)
    merger.config = SimpleNamespace(local_dir=str(tmp_path))

    assert merger._get_world_size() == 1


def test_model_merger_accepts_legacy_global_world_size(tmp_path):
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    with open(tmp_path / "fsdp_config.json", "w") as config_file:
        json.dump({"world_size": 4}, config_file)
    merger = object.__new__(FSDPModelMerger)
    merger.config = SimpleNamespace(local_dir=str(tmp_path))

    assert merger._get_world_size() == 4
