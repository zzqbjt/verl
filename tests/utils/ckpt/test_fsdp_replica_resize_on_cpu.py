"""CPU regression tests for FSDP2 replicated checkpoints resumed on fewer ranks."""

import json
from contextlib import nullcontext

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

from verl.utils.checkpoint.fsdp_checkpoint_manager import (
    FSDPCheckpointManager,
    _rebind_replica_tensor,
)


def _mesh(monkeypatch, world_size, shard_size=1):
    import torch.distributed.device_mesh as mesh_module

    monkeypatch.setattr(mesh_module, "get_rank", lambda: 0)
    return DeviceMesh(
        "cpu", torch.arange(world_size).reshape(-1, shard_size),
        mesh_dim_names=("ddp", "fsdp"), _init_backend=False,
    )


def _tensor(mesh, value):
    local = torch.full((3,), value, dtype=torch.float32)
    spec = DTensorSpec(mesh, (Replicate(), Shard(0)), TensorMeta(local.shape, local.stride(), local.dtype))
    return DTensor(local, spec, requires_grad=False)


class Model(torch.nn.Module):
    def __init__(self, mesh):
        super().__init__()
        self.weight = torch.nn.Parameter(_tensor(mesh, 2.0))
        self.register_buffer("counter", torch.tensor(5))


@pytest.mark.parametrize("new_world_size", [1, 2])
def test_resume_model_adam_scheduler_and_rng(monkeypatch, tmp_path, new_world_size):
    import verl.utils.checkpoint.fsdp_checkpoint_manager as module

    old_mesh = _mesh(monkeypatch, 4)
    new_mesh = _mesh(monkeypatch, new_world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: new_world_size)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(module, "fsdp_version", lambda model: 2)
    monkeypatch.setattr(module, "get_fsdp_state_ctx", lambda *a, **kw: nullcontext())

    old_model = Model(old_mesh)
    old_optimizer = torch.optim.AdamW(old_model.parameters(), lr=0.001, foreach=True)
    old_optimizer.state[old_model.weight] = {
        "step": torch.tensor(50.0),
        "exp_avg": _tensor(old_mesh, 0.2),
        "exp_avg_sq": _tensor(old_mesh, 0.3),
    }
    old_scheduler = torch.optim.lr_scheduler.LambdaLR(old_optimizer, lambda _: 1.0)
    old_scheduler.last_epoch = 50
    (tmp_path / "fsdp_config.json").write_text(json.dumps({
        "FSDP_version": 2, "format_version": 2, "world_size": 4, "shard_world_size": 1,
    }))
    torch.save({"weight": old_model.weight.detach()}, tmp_path / "model_world_size_1_rank_0.pt")
    torch.save(old_optimizer.state_dict(), tmp_path / "optim_world_size_1_rank_0.pt")
    torch.save({"counter": torch.tensor(17)}, tmp_path / "model_buffers_world_size_4_rank_0.pt")
    torch.save({"lr_scheduler": old_scheduler.state_dict(), "rng": {"test": 42}},
               tmp_path / "extra_state_world_size_4_rank_0.pt")
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}

    model = Model(new_mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.9, foreach=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manager = FSDPCheckpointManager(
        model=model, optimizer=optimizer, lr_scheduler=scheduler, device_mesh=new_mesh,
        checkpoint_config={"load_contents": ["model", "optimizer", "extra"], "save_contents": []},
    )
    rng = []
    monkeypatch.setattr(manager, "load_rng_state", rng.append)
    manager.load_checkpoint(str(tmp_path), del_local_after_load=True)
    assert rng == [{"test": 42}]
    assert scheduler.last_epoch == 50
    assert model.counter.item() == 17
    assert optimizer.param_groups[0]["lr"] == 0.001
    state = optimizer.state[model.weight]
    assert state["exp_avg"].device_mesh == new_mesh
    assert state["exp_avg_sq"].device_mesh == new_mesh
    torch.testing.assert_close(state["exp_avg"].to_local(), torch.full((3,), 0.2))
    torch.testing.assert_close(state["exp_avg_sq"].to_local(), torch.full((3,), 0.3))

    # Match a non-distributed AdamW step with exactly the same saved moments.
    reference = torch.nn.Parameter(torch.full((3,), 2.0))
    reference_optimizer = torch.optim.AdamW([reference], lr=0.001, foreach=True)
    reference_optimizer.state[reference] = {
        "step": torch.tensor(50.0), "exp_avg": torch.full((3,), 0.2),
        "exp_avg_sq": torch.full((3,), 0.3),
    }
    model.weight.grad = _tensor(new_mesh, 0.1)
    reference.grad = torch.full((3,), 0.1)
    optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(model.weight.to_local(), reference)
    assert state["step"].item() == 51
    assert before == {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}


def test_reject_actual_shards(monkeypatch):
    saved = _tensor(_mesh(monkeypatch, 4, 2), 1.0)
    target = _tensor(_mesh(monkeypatch, 2), 1.0)
    with pytest.raises(ValueError, match="fsdp_size=1"):
        _rebind_replica_tensor(saved, target)


@pytest.mark.parametrize("saved_shards,current_world", [(2, 2), (1, 8)])
def test_reject_unsupported_resize_before_loading(monkeypatch, tmp_path, saved_shards, current_world):
    import verl.utils.checkpoint.fsdp_checkpoint_manager as module

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: current_world)
    monkeypatch.setattr(module, "fsdp_version", lambda model: 2)
    mesh = _mesh(monkeypatch, current_world)
    manager = FSDPCheckpointManager(
        model=Model(mesh), device_mesh=mesh,
        checkpoint_config={"load_contents": ["model"], "save_contents": []},
    )
    (tmp_path / "fsdp_config.json").write_text(json.dumps({
        "FSDP_version": 2, "format_version": 2, "world_size": 4, "shard_world_size": saved_shards,
    }))
    with pytest.raises(ValueError, match="Changing checkpoint world size"):
        manager.load_checkpoint(str(tmp_path))
