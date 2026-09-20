"""CPU regression coverage for packed PRIME scoring and batch-local reference reuse."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from recipe.prime import prime_dp_rm as module


class ToyModel(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8, generator=torch.Generator().manual_seed(seed)))
        self.calls = 0
        self.tokens = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        self.tokens += input_ids.numel()
        return SimpleNamespace(logits=self.weight[input_ids])


@pytest.fixture(autouse=True)
def cpu_ops(monkeypatch):
    monkeypatch.setattr(module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(module.verl_F, "logprobs_from_logits", lambda logits, labels:
                        logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1))


def make_data(dynamic=False):
    ids = torch.tensor([[0, 1, 2, 3, 0, 0], [1, 2, 3, 4, 5, 6],
                        [0, 3, 4, 5, 6, 0], [1, 4, 5, 6, 0, 0], [0, 5, 6, 7, 1, 0]])
    mask = ids.ne(0).long()
    return DataProto.from_dict(tensors={
        "input_ids": ids, "prompts": ids[:, :2], "responses": ids[:, 2:],
        "attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
        "acc": torch.tensor([0., 1., 0., 1., 1.]),
    }, meta_info={"micro_batch_size": 1, "max_token_len": 12, "use_dynamic_bsz": dynamic})


def make_rm(dynamic=False, packed=True):
    cfg = OmegaConf.create({
        "mini_batch_size": 3, "micro_batch_size_per_gpu": 1, "use_dynamic_bsz": dynamic,
        "ppo_max_token_len_per_gpu": 12, "forward_max_token_len_per_gpu": 12,
        "prime_norm": "batch_norm", "prime_granularity": "token", "lambda": 0.,
        "model": {"use_remove_padding": packed, "use_fused_kernels": False,
                  "loss_type": "ce", "beta_train": .05, "optim": {"grad_clip": 10.}},
    })
    model, ref = ToyModel(3), ToyModel(7).requires_grad_(False)
    return module.DataParallelPRIMERewardModel(cfg, model, ref, torch.optim.SGD(model.parameters(), lr=.1))


def test_packing_preserves_active_log_probs_and_gradients():
    packed, padded = make_rm(), make_rm(packed=False)
    batch = make_data().batch
    mask = batch["attention_mask"][:, 2:].bool()
    a = packed._model_log_probs(packed.reward_module, batch, 2)
    b = padded._model_log_probs(padded.reward_module, batch, 2)
    torch.testing.assert_close(a[mask], b[mask])
    a[mask].sum().backward()
    b[mask].sum().backward()
    torch.testing.assert_close(packed.reward_module.weight.grad, padded.reward_module.weight.grad)
    assert packed.reward_module.tokens == batch["attention_mask"].sum().item()
    assert padded.reward_module.tokens == batch["input_ids"].numel()


def test_reference_cache_reused_without_changing_update_or_post_scores():
    cached, uncached = make_rm(), make_rm()
    data = make_data()
    cached.cache_reference_log_probs(data)
    calls = cached.ref_module.calls
    assert not data.batch["prime_ref_log_probs"].requires_grad
    cached.update_rm(data)
    uncached.update_rm(make_data())
    a, aq, _ = cached.compute_rm_score(data)
    b, bq, _ = uncached.compute_rm_score(make_data())
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(aq, bq)
    torch.testing.assert_close(cached.reward_module.weight, uncached.reward_module.weight)
    assert cached.ref_module.calls == calls == 5
    assert uncached.ref_module.calls == 10
    cached.cache_reference_log_probs(make_data())
    assert cached.ref_module.calls == 10


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("existing_cache", [False, True])
def test_reference_cache_accepts_locked_input_without_copying_tensors(serialized, existing_cache):
    import pickle

    cached, baseline = make_rm(dynamic=True), make_rm(dynamic=True)
    data = make_data(dynamic=True)
    if existing_cache:
        data.batch["prime_ref_log_probs"] = torch.full_like(data.batch["responses"], 123., dtype=torch.float32)
    if serialized:
        # Exercise DataProto's consolidate/serialize/deserialize path used by Ray.
        data = pickle.loads(pickle.dumps(data))
    # Test the locked state observed at the worker cache insertion point.
    data.batch.lock_()
    original = data.batch
    assert original.is_locked
    original_keys = set(original.keys())
    original_values = {key: value.clone() for key, value in original.items()}

    cached.cache_reference_log_probs(data)
    assert data.batch is not original
    assert original.is_locked
    assert set(original.keys()) == original_keys
    for key in original_keys:
        torch.testing.assert_close(original[key], original_values[key])
        if key != "prime_ref_log_probs":
            assert data.batch[key].data_ptr() == original[key].data_ptr()
    assert not data.batch["prime_ref_log_probs"].requires_grad

    cached.update_rm(data)
    baseline.update_rm(make_data(dynamic=True))
    calls = cached.ref_module.calls
    scores, q, _ = cached.compute_rm_score(data)
    expected_scores, expected_q, _ = baseline.compute_rm_score(make_data(dynamic=True))
    assert cached.ref_module.calls == calls
    torch.testing.assert_close(cached.reward_module.weight, baseline.reward_module.weight)
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(q, expected_q)


def test_combined_worker_updates_before_scoring_and_loads_models_once(monkeypatch):
    import inspect
    from contextlib import nullcontext
    from recipe.prime import prime_fsdp_workers as workers

    events = []
    monkeypatch.setattr(workers, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(workers, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(workers.torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(workers.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(workers, "load_fsdp_model_to_gpu", lambda model: events.append("load"))
    monkeypatch.setattr(workers, "offload_fsdp_model_to_cpu", lambda model: events.append("offload"))
    monkeypatch.setattr(workers, "compute_dpo_accuracy", lambda *a, **kw: torch.tensor(.5))
    monkeypatch.setattr(workers, "compute_dpo_abs_accuracy", lambda *a, **kw: torch.tensor(.5))

    class Manager(nullcontext):
        def preprocess_data(self, data):
            return data

        def postprocess_data(self, data):
            return data

    class RM:
        def cache_reference_log_probs(self, data):
            events.append("cache")

        def update_rm(self, data):
            events.append("update")
            return torch.zeros(5, 4), {}

        def compute_rm_score(self, data):
            events.append("score")
            return torch.ones(5, 4), torch.ones(5, 4), {}

    worker = SimpleNamespace(
        _is_offload_param=True, _is_offload_optimizer=False,
        ref_module=object(), reward_module=object(), rm=RM(),
        config=make_rm().config, ulysses_sharding_manager=Manager(),
        reward_lr_scheduler=SimpleNamespace(step=lambda: None, get_last_lr=lambda: [.1]),
    )
    data = make_data()
    data.meta_info.update(n=1, prime_score_after_update=True)
    result = inspect.unwrap(workers.PRIMERewardModelWorker.update_rm)(worker, data)
    assert events == ["load", "load", "cache", "update", "score", "offload", "offload"]
    torch.testing.assert_close(result.batch["rm_scores"], torch.ones(5, 4))


@pytest.mark.parametrize("mock_reorder", [True, False])
def test_dynamic_uneven_batches_restore_rows_and_sample_weighting(monkeypatch, mock_reorder):
    def rearrange(batch, **kwargs):
        # Deliberately reorder rows and use unequal micro-batch sizes.
        rows = list(reversed(range(len(batch))))
        indices = [rows[::2], rows[1::2]]
        indices = [x for x in indices if x]
        return [batch[x] for x in indices], indices

    if mock_reorder:
        monkeypatch.setattr(module, "rearrange_micro_batches", rearrange)
    dynamic, fixed = make_rm(dynamic=True), make_rm()
    data = make_data(dynamic=True)
    dynamic.cache_reference_log_probs(data)
    fixed_data = make_data()
    fixed.cache_reference_log_probs(fixed_data)
    mask = data.batch["attention_mask"][:, 2:].bool()
    torch.testing.assert_close(data.batch["prime_ref_log_probs"][mask], fixed_data.batch["prime_ref_log_probs"][mask])
    a, _ = dynamic.update_rm(data)
    b, _ = fixed.update_rm(fixed_data)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(dynamic.reward_module.weight, fixed.reward_module.weight)
    a, aq, _ = dynamic.compute_rm_score(data)
    b, bq, _ = fixed.compute_rm_score(fixed_data)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(aq, bq)
