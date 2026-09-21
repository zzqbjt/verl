"""CPU-only checks of PRIME's memory handoff to the actor and vLLM."""

import inspect
from types import SimpleNamespace

import pytest

from recipe.prime import prime_fsdp_workers as workers


@pytest.mark.parametrize("version", [0, 1, 2])
@pytest.mark.parametrize("offload_param", [False, True])
@pytest.mark.parametrize("offload_optimizer", [False, True])
def test_release_reshards_before_offload_and_clears_cache_last(monkeypatch, version, offload_param, offload_optimizer):
    events = []

    def model(name):
        return SimpleNamespace(name=name, reshard=lambda: events.append(f"reshard:{name}"))

    worker = SimpleNamespace(
        reward_module=model("rm"),
        ref_module=model("ref"),
        reward_optimizer=object(),
        _is_offload_param=offload_param,
        _is_offload_optimizer=offload_optimizer,
    )
    monkeypatch.setattr(workers, "fsdp_version", lambda model: version)
    monkeypatch.setattr(workers, "reshard_fsdp1_root", lambda model: events.append(f"reshard:{model.name}"))

    def offload_model(model, empty_cache=True):
        # Do not clear the allocator until the optimizer has also been moved.
        assert empty_cache is False
        events.append(f"offload:{model.name}")

    def offload_optimizer_state(optimizer):
        assert optimizer is worker.reward_optimizer
        events.append("offload:optimizer")

    monkeypatch.setattr(workers, "offload_fsdp_model_to_cpu", offload_model)
    monkeypatch.setattr(workers, "offload_fsdp_optimizer", offload_optimizer_state)
    monkeypatch.setattr(workers.torch.distributed, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(
        workers,
        "get_torch_device",
        lambda: SimpleNamespace(
            synchronize=lambda: events.append("synchronize"),
            empty_cache=lambda: events.append("empty_cache"),
        ),
    )

    workers.PRIMERewardModelWorker._release_after_use(worker)
    expected = ["reshard:rm", "reshard:ref"] if version else []
    if offload_param:
        expected += ["offload:rm", "offload:ref"]
    if offload_optimizer:
        expected += ["offload:optimizer"]
    if offload_param or offload_optimizer:
        expected += ["synchronize", "barrier", "empty_cache"]
    assert events == expected


@pytest.mark.parametrize("method", ["save_checkpoint", "load_checkpoint"])
@pytest.mark.parametrize("offload_param", [False, True])
def test_checkpoint_hands_off_memory_after_io(monkeypatch, method, offload_param):
    events = []
    model = object()

    def load_model(value):
        assert value is model
        events.append("load_model")

    def checkpoint_io(**kwargs):
        assert kwargs["local_path"] == "test-checkpoint"
        events.append("checkpoint_io")

    monkeypatch.setattr(workers, "load_fsdp_model_to_gpu", load_model)
    monkeypatch.setattr(workers.torch.distributed, "barrier", lambda: events.append("barrier"))
    worker = SimpleNamespace(
        reward_module=model,
        _is_offload_param=offload_param,
        checkpoint_manager=SimpleNamespace(**{method: checkpoint_io}),
        _release_after_use=lambda: events.append("release_models_and_optimizer"),
    )
    inspect.unwrap(getattr(workers.PRIMERewardModelWorker, method))(worker, "test-checkpoint")
    expected = ["load_model"] if offload_param else []
    assert events == expected + ["checkpoint_io", "barrier", "release_models_and_optimizer"]
