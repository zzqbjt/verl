from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.workers.config.optimizer import build_optimizer


def test_prime_dapo_optimizer_supports_struct_config_and_cpu_step(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    with initialize_config_dir(config_dir=str(root / "recipe/dapo/config"), version_base=None):
        config = compose(config_name="prime_dapo_trainer")
    OmegaConf.set_struct(config, True)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = build_optimizer([parameter], config.prime.model.optim)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert tuple(optimizer.defaults["betas"]) == (0.9, 0.999)
    assert optimizer.defaults["lr"] == config.prime.model.optim.lr
    assert optimizer.defaults["weight_decay"] == config.prime.model.optim.weight_decay
    parameter.square().sum().backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert parameter.item() < 1.0
