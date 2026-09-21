from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.utils.torch_dtypes import PrecisionType
from verl.workers.config.optimizer import build_optimizer


@pytest.fixture
def prime_config(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    with initialize_config_dir(config_dir=str(root / "recipe/dapo/config"), version_base=None):
        config = compose(config_name="prime_dapo_trainer")
    OmegaConf.set_struct(config, True)
    return config.prime


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_prime_inherits_actor_strategy_and_trainer_accepts_it(monkeypatch, strategy):
    from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
    from recipe.dapo.prime_dapo_ray_trainer import RayPrimeDAPOTrainer

    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    with initialize_config_dir(config_dir=str(root / "recipe/dapo/config"), version_base=None):
        config = compose(config_name="prime_dapo_trainer", overrides=[f"actor_rollout_ref.actor.strategy={strategy}"])
    assert config.prime.strategy == strategy

    def init_base(self, **kwargs):
        self.config = kwargs["config"]
        self.use_legacy_worker_impl = "enable"
        self.total_training_steps = 10

    monkeypatch.setattr(RayDAPOTrainer, "__init__", init_base)
    trainer = RayPrimeDAPOTrainer(config=config)
    assert trainer.config.prime.model.optim.total_training_steps == 10


def test_prime_dapo_optimizer_supports_struct_config_and_cpu_step(prime_config):
    dtype = PrecisionType.to_dtype(prime_config.model.fsdp_config.model_dtype)
    assert dtype == torch.float32
    parameter = torch.nn.Parameter(torch.tensor([0.001, 0.01, 0.02, 0.1, 1.0], dtype=dtype))
    before = parameter.detach().clone()
    optimizer = build_optimizer([parameter], prime_config.model.optim)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert tuple(optimizer.defaults["betas"]) == (0.9, 0.999)
    assert optimizer.defaults["lr"] == prime_config.model.optim.lr
    assert optimizer.defaults["weight_decay"] == prime_config.model.optim.weight_decay
    parameter.sum().backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert torch.all(parameter < before)
    assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
    assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_prime_worker_keeps_fp32_trainable_weights_and_bf16_compute(prime_config, monkeypatch, strategy):
    import torch.distributed.fsdp as fsdp
    import transformers
    from torch.distributed.fsdp import ShardingStrategy

    from recipe.prime import prime_fsdp_workers as worker_module

    loaded_dtypes = []
    fsdp_options = []
    restored_dtypes = []
    prime_config.strategy = strategy

    class ToyModel(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, dtype=dtype))

        def gradient_checkpointing_enable(self, **kwargs):
            pass

    def load_model(**kwargs):
        loaded_dtypes.append(kwargs["torch_dtype"])
        return ToyModel(kwargs["torch_dtype"])

    def wrap_model(model, **kwargs):
        fsdp_options.append(kwargs)
        return model

    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(tie_word_embeddings=True)
    )
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(worker_module, "copy_local_path_from_hdfs", lambda path: path)
    monkeypatch.setattr(
        worker_module,
        "hf_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(bos_token_id=0, eos_token_id=1, pad_token_id=2),
    )
    monkeypatch.setattr(worker_module, "get_init_weight_context_manager", lambda **kwargs: nullcontext)
    monkeypatch.setattr(worker_module, "apply_monkey_patch", lambda **kwargs: None)
    monkeypatch.setattr(worker_module, "get_fsdp_wrap_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(worker_module, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(fsdp, "FullyShardedDataParallel", wrap_model)
    monkeypatch.setattr(worker_module, "apply_fsdp2", lambda model, kwargs, config: fsdp_options.append(kwargs))
    monkeypatch.setattr(
        worker_module,
        "fsdp2_load_full_state_dict",
        lambda model, state, mesh, **kwargs: restored_dtypes.append(state["weight"].dtype),
    )
    mesh = MagicMock(ndim=2, mesh_dim_names=("ddp", "fsdp"))
    mesh.size.return_value = 1
    mesh.shape = (4, 1)
    replica_mesh = mesh["ddp"]
    worker = SimpleNamespace(config=prime_config, rank=1, device_mesh=mesh, ulysses_sequence_parallel_size=1)
    worker._wrap_fsdp_model = lambda model, mp: worker_module.PRIMERewardModelWorker._wrap_fsdp_model(worker, model, mp)
    model, ref, optimizer, _ = worker_module.PRIMERewardModelWorker._build_reward_ref_model_optimizer(
        worker, prime_config
    )

    assert loaded_dtypes == [torch.float32, torch.bfloat16]
    assert model.weight.dtype == torch.float32
    assert ref.weight.dtype == torch.bfloat16
    assert model.weight.requires_grad
    assert not ref.weight.requires_grad
    assert not ref.training
    assert len(fsdp_options) == 2
    for options in fsdp_options:
        if strategy == "fsdp":
            assert options["sharding_strategy"] == ShardingStrategy.NO_SHARD
            assert options["device_mesh"] is replica_mesh
            mixed_precision = options["mixed_precision"]
            assert mixed_precision.buffer_dtype == torch.float32
        else:
            assert options["mesh"] is mesh
            assert options["reshard_after_forward"] is True
            assert options["offload_policy"] is None
            mixed_precision = options["mp_policy"]
        assert mixed_precision.param_dtype == torch.bfloat16
        assert mixed_precision.reduce_dtype == torch.float32
    assert restored_dtypes == ([torch.float32, torch.bfloat16] if strategy == "fsdp2" else [])

    model.weight.sum().backward()
    optimizer.step()
    assert torch.all(model.weight < 1.0)
    assert optimizer.state[model.weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[model.weight]["exp_avg_sq"].dtype == torch.float32


def test_loading_bf16_checkpoint_preserves_fp32_parameters_and_adam_states(prime_config):
    # Exercise the same load_state_dict calls as FSDPCheckpointManager without
    # requiring a GPU process group or loading a real multi-GB checkpoint.
    old_model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    old_optimizer = build_optimizer(old_model.parameters(), prime_config.model.optim)
    old_model.weight.sum().backward()
    old_optimizer.step()
    assert old_optimizer.state[old_model.weight]["exp_avg"].dtype == torch.bfloat16

    dtype = PrecisionType.to_dtype(prime_config.model.fsdp_config.model_dtype)
    model = torch.nn.Linear(2, 1, bias=False, dtype=dtype)
    optimizer = build_optimizer(model.parameters(), prime_config.model.optim)
    model.load_state_dict(old_model.state_dict())
    optimizer.load_state_dict(old_optimizer.state_dict())

    assert model.weight.dtype == torch.float32
    for key in ("exp_avg", "exp_avg_sq"):
        assert optimizer.state[model.weight][key].dtype == torch.float32
        torch.testing.assert_close(
            optimizer.state[model.weight][key], old_optimizer.state[old_model.weight][key].float()
        )
    before = model.weight.detach().clone()
    model.weight.sum().backward()
    optimizer.step()
    assert torch.all(model.weight < before)
