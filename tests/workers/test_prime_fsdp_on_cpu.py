"""Real CPU/Gloo coverage for PRIME FSDP wrapping, synchronization and resume."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import PretrainedConfig

from verl.utils.fsdp_utils import get_fsdp1_wrap_kwargs, reshard_fsdp1_root


@pytest.mark.parametrize("sharded", [False, True])
def test_reshard_root_only_releases_sharded_storage(sharded):
    handle = MagicMock(uses_sharded_strategy=sharded)
    reshard_fsdp1_root(SimpleNamespace(_handle=handle))
    if sharded:
        handle.reshard.assert_called_once_with(True)
    else:
        handle.reshard.assert_not_called()
    reshard_fsdp1_root(SimpleNamespace(_handle=None))


@pytest.mark.parametrize("shape", [(4,), (2, 2), (4, 1)])
def test_fsdp1_mesh_conversion_only_changes_singleton_shards(shape):
    mesh = MagicMock(ndim=len(shape), mesh_dim_names=("ddp", "fsdp"))
    mesh.size.side_effect = lambda dim: shape[dim]
    strategy = ShardingStrategy.HYBRID_SHARD if len(shape) == 2 else ShardingStrategy.FULL_SHARD
    kwargs = get_fsdp1_wrap_kwargs(mesh, strategy)
    if shape == (4, 1):
        assert kwargs["device_mesh"] is mesh["ddp"]
        assert kwargs["sharding_strategy"] == ShardingStrategy.NO_SHARD
    else:
        assert kwargs["device_mesh"] is mesh
        assert kwargs["sharding_strategy"] == strategy


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.125, 0.25]]))

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight)


class ToyRewardModel(torch.nn.Module):
    _no_split_modules = ["ToyBlock"]

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.config = PretrainedConfig(tie_word_embeddings=True)
        self.layer = ToyBlock().to(dtype)

    def forward(self, inputs):
        return self.layer(inputs)

    def can_generate(self):
        return False


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _distributed_prime_worker(rank, rendezvous, checkpoint_dir):
    from recipe.prime.prime_dp_rm import DataParallelPRIMERewardModel
    from recipe.prime.prime_fsdp_workers import PRIMERewardModelWorker
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    from verl.workers.config.optimizer import build_optimizer

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("ddp", "fsdp"))
        for strategy in ("fsdp", "fsdp2"):
            config = OmegaConf.create(
                {
                    "strategy": strategy,
                    "model": {
                        "fsdp_config": {"fsdp_size": 1, "wrap_policy": {}},
                        "optim": {
                            "optimizer": "AdamW",
                            "optimizer_impl": "torch.optim",
                            "lr": 1e-6,
                            "weight_decay": 0.0,
                            "betas": [0.9, 0.999],
                            "override_optimizer_config": None,
                            "grad_clip": 1.0,
                        },
                    },
                }
            )
            worker = SimpleNamespace(config=config, device_mesh=mesh)
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            )

            def cpu_fsdp(model, **kwargs):
                # FSDP1's initial sync_module_states broadcast requires CUDA.
                # The deterministic toy starts identically; backward uses real Gloo.
                kwargs["sync_module_states"] = False
                return FSDP(model, **kwargs)

            with patch("torch.distributed.fsdp.FullyShardedDataParallel", cpu_fsdp):
                model = PRIMERewardModelWorker._wrap_fsdp_model(worker, ToyRewardModel(), mixed_precision)
                ref = PRIMERewardModelWorker._wrap_fsdp_model(
                    worker, ToyRewardModel(torch.bfloat16).requires_grad_(False).eval(), mixed_precision
                )

            if strategy == "fsdp":
                for module in FSDP.fsdp_modules(model):
                    assert module.sharding_strategy == ShardingStrategy.NO_SHARD
                    assert dist.get_world_size(module.process_group) == 2
            else:
                for parameter in model.parameters():
                    assert parameter.device_mesh.shape == (2, 1)

            optimizer = build_optimizer(model.parameters(), config.model.optim)
            rm = DataParallelPRIMERewardModel(config, model, ref, optimizer)
            baseline = ToyRewardModel()
            baseline_optimizer = build_optimizer(baseline.parameters(), config.model.optim)
            local_input = torch.tensor([[float(rank + 1), 2.0]])
            all_inputs = torch.tensor([[1.0, 2.0], [2.0, 2.0]])
            reference_before = [_local(p).detach().clone() for p in ref.parameters()]

            for _ in range(3):
                # Score-before-training and frozen-reference inference must be repeatable.
                with torch.no_grad():
                    model(local_input)
                    ref(local_input)
                if strategy == "fsdp":
                    reshard_fsdp1_root(model)
                else:
                    model.reshard()
                    ref.reshard()

                optimizer.zero_grad()
                model(local_input).sum().backward()
                baseline_optimizer.zero_grad()
                baseline(all_inputs).mean().backward()
                for actual, expected in zip(model.parameters(), baseline.parameters(), strict=True):
                    torch.testing.assert_close(_local(actual.grad).reshape_as(expected.grad), expected.grad)

                expected_norm = torch.nn.utils.clip_grad_norm_(baseline.parameters(), max_norm=1.0)
                actual_norm = rm._optimizer_step()
                baseline_optimizer.step()
                torch.testing.assert_close(actual_norm, expected_norm)
                for actual, expected in zip(model.parameters(), baseline.parameters(), strict=True):
                    assert actual.dtype == torch.float32
                    value = _local(actual).detach()
                    torch.testing.assert_close(value.reshape_as(expected), expected, atol=1e-8, rtol=0)
                    peers = [torch.empty_like(value) for _ in range(2)]
                    dist.all_gather(peers, value)
                    torch.testing.assert_close(peers[0], peers[1], atol=0, rtol=0)
                    assert optimizer.state[actual]["exp_avg"].dtype == torch.float32
                    assert optimizer.state[actual]["exp_avg_sq"].dtype == torch.float32

            for parameter, before in zip(ref.parameters(), reference_before, strict=True):
                assert not parameter.requires_grad
                assert parameter.dtype == torch.bfloat16
                torch.testing.assert_close(_local(parameter), before, atol=0, rtol=0)

            if strategy == "fsdp2":
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
                manager = FSDPCheckpointManager(
                    model,
                    optimizer,
                    lr_scheduler=scheduler,
                    device_mesh=mesh,
                    checkpoint_config={
                        "save_contents": ["model", "optimizer"],
                        "load_contents": ["model", "optimizer"],
                    },
                )
                saved = [_local(parameter).detach().clone() for parameter in model.parameters()]
                saved_moments = [_local(optimizer.state[p]["exp_avg"]).clone() for p in model.parameters()]
                manager.save_checkpoint(checkpoint_dir, global_step=3)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(1.0)
                        optimizer.state[parameter]["exp_avg"].zero_()
                manager.load_checkpoint(checkpoint_dir)
                for parameter, expected, moment in zip(model.parameters(), saved, saved_moments, strict=True):
                    torch.testing.assert_close(_local(parameter), expected, atol=0, rtol=0)
                    torch.testing.assert_close(_local(optimizer.state[parameter]["exp_avg"]), moment, atol=0, rtol=0)
                # Resume one more real update and compare against uninterrupted training.
                optimizer.zero_grad()
                model(local_input).sum().backward()
                rm._optimizer_step()
                baseline_optimizer.zero_grad()
                baseline(all_inputs).mean().backward()
                torch.nn.utils.clip_grad_norm_(baseline.parameters(), max_norm=1.0)
                baseline_optimizer.step()
                for actual, expected in zip(model.parameters(), baseline.parameters(), strict=True):
                    torch.testing.assert_close(_local(actual).reshape_as(expected), expected, atol=1e-8, rtol=0)
    finally:
        dist.destroy_process_group()


def test_prime_fsdp_two_rank_cpu_sync_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(
        _distributed_prime_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "checkpoint")),
        nprocs=2,
        join=True,
    )
    assert (tmp_path / "checkpoint" / "model_world_size_1_rank_0.pt").exists()
    assert not (tmp_path / "checkpoint" / "model_world_size_2_rank_1.pt").exists()
