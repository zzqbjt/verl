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

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from verl import DataProto
from verl.trainer.config import DapoReferenceKLConfig
from verl.trainer.ppo.dapo_reference_kl import prepare_dapo_reference_kl_inputs
from verl.workers.actor.dp_actor import (
    DataParallelPPOActor,
    dapo_reference_topk_forward_kl,
    gather_dapo_reference_teacher_topk,
    mix_dapo_reference_kl_loss,
    summarize_dapo_reference_teacher_topk,
)


def test_dapo_reference_kl_config_defaults_and_validation():
    config = DapoReferenceKLConfig()
    assert config.enabled is False
    assert config.loss_coef == 0.0
    assert config.approximation == "topk"
    assert config.top_k == 100
    assert config.token_chunk_size == 512

    with pytest.raises(ValueError, match="loss_coef"):
        DapoReferenceKLConfig(loss_coef=-0.1)
    with pytest.raises(ValueError, match="loss_coef"):
        DapoReferenceKLConfig(loss_coef=1.1)
    with pytest.raises(ValueError, match="approximation"):
        DapoReferenceKLConfig(approximation="sampled")
    with pytest.raises(ValueError, match="approximation"):
        DapoReferenceKLConfig(approximation="exact")
    with pytest.raises(ValueError, match="token_chunk_size"):
        DapoReferenceKLConfig(token_chunk_size=0)
    with pytest.raises(ValueError, match="token_chunk_size"):
        DapoReferenceKLConfig(token_chunk_size=True)
    with pytest.raises(ValueError, match="reference_solution"):
        DapoReferenceKLConfig(teacher_prompt_template="{question}")


@pytest.mark.parametrize("loss_coef", [0.0, 0.1, 1.0])
def test_mix_dapo_reference_kl_loss_is_convex_combination(loss_coef):
    dapo_loss = torch.tensor(2.0, requires_grad=True)
    reference_kl_loss = torch.tensor(5.0, requires_grad=True)

    actual = mix_dapo_reference_kl_loss(dapo_loss, reference_kl_loss, loss_coef)
    expected = (1.0 - loss_coef) * dapo_loss + loss_coef * reference_kl_loss
    torch.testing.assert_close(actual, expected)

    actual.backward()
    torch.testing.assert_close(dapo_loss.grad, torch.tensor(1.0 - loss_coef))
    torch.testing.assert_close(reference_kl_loss.grad, torch.tensor(loss_coef))


class _RecordingTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.teacher_texts = []

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "solution:" + ",".join(str(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is True
        assert add_generation_prompt is True
        text = messages[0]["content"]
        self.teacher_texts.append(text)
        return [10 + (ord(char) % 31) for char in text]


def _make_reference_batch():
    responses = torch.tensor(
        [
            [11, 12, 1, 0],
            [21, 22, 1, 0],
            [31, 1, 0, 0],
            [41, 42, 1, 0],
            [51, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    return DataProto.from_single_dict(
        {
            "responses": responses,
            "response_mask": response_mask,
            "raw_prompt": np.array(
                [[{"role": "user", "content": "fallback question"}]] * 5,
                dtype=object,
            ),
            "extra_info": np.array(
                [
                    {"question": "question a"},
                    {"question": "question a"},
                    {"question": "question a"},
                    {"question": "question b"},
                    {"question": "question b"},
                ],
                dtype=object,
            ),
            "uid": np.array(["group-a", "group-a", "group-a", "group-b", "group-b"], dtype=object),
            "acc": np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=object),
        }
    )


def test_prepare_dapo_reference_kl_uses_correct_rollout_only_for_incorrect_rows():
    tokenizer = _RecordingTokenizer()
    batch = _make_reference_batch()
    metrics = prepare_dapo_reference_kl_inputs(
        batch,
        tokenizer,
        DapoReferenceKLConfig(enabled=True, loss_coef=0.1, max_teacher_prompt_length=1024),
        metric_key="acc",
    )

    torch.testing.assert_close(
        batch.batch["dapo_reference_kl_reference_rows"],
        torch.tensor([-1, 0, 0, -1, 3]),
    )
    expected_loss_mask = torch.zeros_like(batch.batch["response_mask"])
    expected_loss_mask[1:3] = batch.batch["response_mask"][1:3]
    expected_loss_mask[4] = batch.batch["response_mask"][4]
    torch.testing.assert_close(batch.batch["dapo_reference_kl_loss_mask"], expected_loss_mask)

    assert len(tokenizer.teacher_texts) == 3
    assert all("question a" in text for text in tokenizer.teacher_texts[:2])
    assert all(
        "Here is a reference solution: solution:11,12,1" in text for text in tokenizer.teacher_texts[:2]
    )
    assert "question b" in tokenizer.teacher_texts[2]
    assert "Here is a reference solution: solution:41,42,1" in tokenizer.teacher_texts[2]
    assert metrics["dapo_reference_kl/incorrect_responses"] == 3.0

    response_length = batch.batch["responses"].shape[-1]
    torch.testing.assert_close(
        batch.batch["dapo_reference_teacher_input_ids"][:, -response_length:],
        batch.batch["responses"],
    )
    torch.testing.assert_close(
        batch.batch["dapo_reference_teacher_attention_mask"][:, -response_length:],
        batch.batch["response_mask"],
    )


def test_prepare_dapo_reference_kl_requires_dapo_metric():
    batch = _make_reference_batch()
    batch.non_tensor_batch.pop("acc")
    with pytest.raises(ValueError, match="non-tensor batch keys.*acc"):
        prepare_dapo_reference_kl_inputs(
            batch,
            _RecordingTokenizer(),
            DapoReferenceKLConfig(enabled=True),
            metric_key="acc",
        )


def test_prepare_dapo_reference_kl_requires_dapo_filtered_groups():
    batch = _make_reference_batch()
    batch.non_tensor_batch["acc"][4] = 1.0
    with pytest.raises(ValueError, match="already retained by DAPO dynamic sampling"):
        prepare_dapo_reference_kl_inputs(
            batch,
            _RecordingTokenizer(),
            DapoReferenceKLConfig(enabled=True),
            metric_key="acc",
        )


def test_prepare_dapo_reference_kl_requires_binary_dapo_metric():
    batch = _make_reference_batch()
    batch.non_tensor_batch["acc"][1] = 0.5
    with pytest.raises(ValueError, match="expects binary 'acc' values"):
        prepare_dapo_reference_kl_inputs(
            batch,
            _RecordingTokenizer(),
            DapoReferenceKLConfig(enabled=True),
            metric_key="acc",
        )


@pytest.mark.parametrize("top_k", [1, 4, 13])
def test_dapo_reference_topk_forward_kl_matches_coarsened_distribution(top_k):
    generator = torch.Generator().manual_seed(11)
    student = torch.randn(6, 13, generator=generator, requires_grad=True)
    teacher = torch.randn(6, 13, generator=generator)

    indices, teacher_top_log_probs, teacher_tail_prob = summarize_dapo_reference_teacher_topk(
        teacher, top_k
    )
    actual = dapo_reference_topk_forward_kl(
        student,
        teacher_top_indices=indices,
        teacher_top_log_probs=teacher_top_log_probs,
        teacher_tail_prob=teacher_tail_prob,
        token_chunk_size=2,
    )

    teacher_probs = torch.softmax(teacher.float(), dim=-1)
    student_probs = torch.softmax(student.float(), dim=-1)
    teacher_top_probs = teacher_probs.gather(-1, indices)
    student_top_probs = student_probs.gather(-1, indices)
    expected = (
        teacher_top_probs * (teacher_top_probs.log() - student_top_probs.log())
    ).sum(dim=-1)
    if top_k < teacher.shape[-1]:
        expected = expected + teacher_tail_prob * (
            teacher_tail_prob.log() - (1.0 - student_top_probs.sum(dim=-1)).log()
        )

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual.mean().backward()
    assert torch.isfinite(student.grad).all()


def test_dapo_reference_topk_forward_kl_chunking_preserves_gradients():
    generator = torch.Generator().manual_seed(37)
    student_chunked = torch.randn(7, 19, generator=generator, requires_grad=True)
    student_unchunked = student_chunked.detach().clone().requires_grad_(True)
    teacher = torch.randn(7, 19, generator=generator)
    token_weights = torch.randn(7, generator=generator)

    indices, teacher_top_log_probs, teacher_tail_prob = summarize_dapo_reference_teacher_topk(
        teacher, top_k=5
    )
    chunked = dapo_reference_topk_forward_kl(
        student_chunked,
        teacher_top_indices=indices,
        teacher_top_log_probs=teacher_top_log_probs,
        teacher_tail_prob=teacher_tail_prob,
        token_chunk_size=3,
    )
    unchunked = dapo_reference_topk_forward_kl(
        student_unchunked,
        teacher_top_indices=indices,
        teacher_top_log_probs=teacher_top_log_probs,
        teacher_tail_prob=teacher_tail_prob,
        token_chunk_size=student_unchunked.shape[0],
    )

    torch.testing.assert_close(chunked, unchunked, atol=2e-6, rtol=2e-6)
    (chunked * token_weights).sum().backward()
    (unchunked * token_weights).sum().backward()
    torch.testing.assert_close(student_chunked.grad, student_unchunked.grad, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("token_chunk_size", [0, -1, True, 1.5])
def test_dapo_reference_topk_forward_kl_rejects_invalid_chunk_size(token_chunk_size):
    student = torch.randn(2, 5)
    teacher = torch.randn(2, 5)
    indices, teacher_top_log_probs, teacher_tail_prob = summarize_dapo_reference_teacher_topk(teacher, top_k=2)

    with pytest.raises(ValueError, match="token_chunk_size"):
        dapo_reference_topk_forward_kl(
            student,
            teacher_top_indices=indices,
            teacher_top_log_probs=teacher_top_log_probs,
            teacher_tail_prob=teacher_tail_prob,
            token_chunk_size=token_chunk_size,
        )


def test_gather_dapo_reference_teacher_topk_uses_old_policy_cache_in_micro_batch_order():
    teacher_cache = {
        1: (
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            torch.tensor([[-0.1, -1.0], [-0.2, -1.2]]),
            torch.tensor([0.1, 0.2]),
        ),
        4: (
            torch.tensor([[5, 6]], dtype=torch.int32),
            torch.tensor([[-0.3, -1.3]]),
            torch.tensor([0.3]),
        ),
    }
    indices, log_probs, tail_probs = gather_dapo_reference_teacher_topk(
        teacher_cache=teacher_cache,
        row_ids=torch.tensor([4, 9, 1]),
        loss_mask=torch.tensor([[1, 0], [0, 0], [1, 1]]),
        device="cpu",
    )

    torch.testing.assert_close(indices, torch.tensor([[5, 6], [1, 2], [3, 4]]))
    torch.testing.assert_close(
        log_probs,
        torch.tensor([[-0.3, -1.3], [-0.1, -1.0], [-0.2, -1.2]]),
    )
    torch.testing.assert_close(tail_probs, torch.tensor([0.3, 0.1, 0.2]))


class _TokenIdLogitModel(nn.Module):
    def forward(self, input_ids, **kwargs):
        logits = torch.nn.functional.one_hot(input_ids % 17, num_classes=17).float() * 3
        return SimpleNamespace(logits=logits)


class _CountingTokenIdLogitModel(_TokenIdLogitModel):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, input_ids, **kwargs):
        self.forward_calls += 1
        return super().forward(input_ids, **kwargs)


class _TrainableLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 8)
        self.output = nn.Linear(8, 17, bias=False)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def _make_minimal_actor(*, use_remove_padding, model=None):
    actor = object.__new__(DataParallelPPOActor)
    actor.config = {"calculate_sum_pi_squared": False, "sum_pi_squared_checkpointing": False}
    actor.use_remove_padding = use_remove_padding
    actor.use_ulysses_sp = False
    actor.use_fused_kernels = False
    actor.use_prefix_grouper = False
    actor.use_dynamic_bsz = False
    actor.device_name = "cpu"
    actor.param_dtype = torch.bfloat16
    actor.actor_module = _TokenIdLogitModel() if model is None else model
    return actor


def test_precompute_reference_teacher_targets_uses_one_fixed_old_policy_pass(monkeypatch):
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_id", lambda: "cpu")
    actor = _make_minimal_actor(use_remove_padding=False)
    actor.config = SimpleNamespace(
        use_dynamic_bsz=False,
        ppo_micro_batch_size_per_gpu=1,
        get=lambda key, default=None: default,
    )
    actor.ulysses_sequence_parallel_size = 1
    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[5, 6], [7, 0], [8, 0]]),
            "response_mask": torch.tensor([[1, 1], [1, 0], [1, 0]]),
            "dapo_reference_teacher_input_ids": torch.tensor(
                [[1, 2, 5, 6], [1, 3, 7, 0], [2, 4, 8, 0]]
            ),
            "dapo_reference_teacher_attention_mask": torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0]]
            ),
            "dapo_reference_teacher_position_ids": torch.tensor(
                [[0, 1, 2, 3], [0, 1, 2, 0], [0, 1, 2, 0]]
            ),
            "dapo_reference_kl_loss_mask": torch.tensor([[1, 1], [0, 0], [1, 0]]),
            "dapo_reference_kl_row_id": torch.tensor([10, 11, 12]),
        }
    )

    cache = actor._precompute_dapo_reference_old_policy_topk(
        data=data,
        temperature=1.0,
        top_k=3,
        pad_token_id=0,
    )

    assert set(cache) == {10, 12}
    assert cache[10][0].shape == (2, 3)
    assert cache[12][0].shape == (1, 3)
    assert cache[10][0].dtype == torch.int32
    assert cache[10][1].device.type == "cpu"
    assert cache[10][1].requires_grad is False


def _run_reference_teacher_collective_alignment_rank(rank, world_size, init_method):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    try:
        import verl.workers.actor.dp_actor as dp_actor_module

        dp_actor_module.get_device_id = lambda: "cpu"
        model = _CountingTokenIdLogitModel()
        actor = _make_minimal_actor(use_remove_padding=False, model=model)
        actor.config = SimpleNamespace(
            use_dynamic_bsz=False,
            ppo_micro_batch_size_per_gpu=1,
            get=lambda key, default=None: default,
        )
        actor.ulysses_sequence_parallel_size = 1
        loss_mask = (
            torch.tensor([[1, 1], [1, 0]])
            if rank == 0
            else torch.zeros((2, 2), dtype=torch.long)
        )
        data = DataProto.from_dict(
            tensors={
                "responses": torch.tensor([[5, 6], [7, 0]]),
                "response_mask": torch.tensor([[1, 1], [1, 0]]),
                "dapo_reference_teacher_input_ids": torch.tensor([[1, 2, 5, 6], [1, 3, 7, 0]]),
                "dapo_reference_teacher_attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
                "dapo_reference_teacher_position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]]),
                "dapo_reference_kl_loss_mask": loss_mask,
                "dapo_reference_kl_row_id": torch.tensor([10, 11]),
            }
        )

        cache = actor._precompute_dapo_reference_old_policy_topk(
            data=data,
            temperature=1.0,
            top_k=3,
            pad_token_id=0,
        )

        assert model.forward_calls == 2
        assert set(cache) == ({10, 11} if rank == 0 else set())
    finally:
        dist.destroy_process_group()


def test_precompute_reference_teacher_aligns_forward_count_when_a_rank_has_no_selected_rows(tmp_path):
    init_method = f"file://{tmp_path / 'reference-teacher-dist-init'}"
    mp.start_processes(
        _run_reference_teacher_collective_alignment_rank,
        args=(2, init_method),
        nprocs=2,
        join=True,
        start_method="fork",
    )


@pytest.mark.parametrize("use_remove_padding", [False, True])
def test_forward_micro_batch_returns_policy_outputs_and_selected_kl_logits(use_remove_padding, monkeypatch):
    def _cpu_logprobs(logits, labels, **kwargs):
        return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    monkeypatch.setattr("verl.workers.actor.dp_actor.logprobs_from_logits", _cpu_logprobs)
    model_inputs = {
        "input_ids": torch.tensor([[0, 5, 6, 7, 8, 0], [3, 4, 9, 10, 0, 0]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]),
        "position_ids": torch.tensor([[0, 0, 1, 2, 3, 0], [0, 1, 2, 3, 0, 0]]),
        "responses": torch.tensor([[7, 8, 0], [10, 0, 0]]),
        "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        "dapo_reference_kl_loss_mask": torch.tensor([[0, 1, 0], [1, 0, 0]]),
    }
    outputs = _make_minimal_actor(use_remove_padding=use_remove_padding)._forward_micro_batch(
        model_inputs,
        temperature=1.0,
        return_valid_logits=True,
    )

    assert outputs["log_probs"].shape == model_inputs["response_mask"].shape
    assert outputs["valid_logits"].shape == (2, 17)


def test_policy_log_probs_and_reference_kl_share_one_student_forward_backward(monkeypatch):
    def _cpu_logprobs(logits, labels, **kwargs):
        return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    monkeypatch.setattr("verl.workers.actor.dp_actor.logprobs_from_logits", _cpu_logprobs)
    model = _TrainableLogitModel()
    actor = _make_minimal_actor(use_remove_padding=True, model=model)
    model_inputs = {
        "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
        "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 0, 0]]),
        "responses": torch.tensor([[5, 6, 7], [11, 0, 0]]),
        "response_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
        "dapo_reference_kl_loss_mask": torch.tensor([[0, 1, 1], [1, 0, 0]]),
    }
    outputs = actor._forward_micro_batch(model_inputs, temperature=1.0, return_valid_logits=True)
    teacher_logits = torch.randn_like(outputs["valid_logits"])
    indices, teacher_top_log_probs, teacher_tail_prob = summarize_dapo_reference_teacher_topk(
        teacher_logits, top_k=5
    )
    reference_kl = dapo_reference_topk_forward_kl(
        outputs["valid_logits"],
        teacher_top_indices=indices,
        teacher_top_log_probs=teacher_top_log_probs,
        teacher_tail_prob=teacher_tail_prob,
    )
    policy_loss = -outputs["log_probs"][model_inputs["response_mask"].bool()].mean()
    reference_kl_coef = 0.1
    total_loss = (
        (1.0 - reference_kl_coef) * policy_loss
        + reference_kl_coef * reference_kl.mean()
    )
    total_loss.backward()

    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all()


def test_reference_kl_token_metric_includes_zero_for_an_all_correct_micro_batch(monkeypatch):
    def _cpu_logprobs(logits, labels, **kwargs):
        return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    recorded_chunk_sizes = []

    def _recording_reference_kl(*args, token_chunk_size, **kwargs):
        recorded_chunk_sizes.append(token_chunk_size)
        return dapo_reference_topk_forward_kl(*args, token_chunk_size=token_chunk_size, **kwargs)

    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_id", lambda: "cpu")
    monkeypatch.setattr("verl.workers.actor.dp_actor.logprobs_from_logits", _cpu_logprobs)
    monkeypatch.setattr("verl.workers.actor.dp_actor.dapo_reference_topk_forward_kl", _recording_reference_kl)
    model = _TrainableLogitModel()
    actor = _make_minimal_actor(use_remove_padding=False, model=model)
    actor.config = SimpleNamespace(
        use_dynamic_bsz=False,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        ppo_epochs=1,
        policy_loss={"loss_mode": "vanilla"},
        use_kl_loss=False,
        kl_loss_coef=0.0,
        calculate_entropy=False,
        entropy_coeff=0.0,
        loss_agg_mode="token-mean",
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        global_batch_info={},
        get=lambda key, default=None: default,
    )
    actor.ulysses_sequence_parallel_size = 1
    actor.actor_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    actor.scaler = None
    actor._optimizer_step = lambda: torch.tensor(0.0)
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2, 5, 6], [1, 3, 7, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
            "position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]]),
            "responses": torch.tensor([[5, 6], [7, 0]]),
            "response_mask": torch.tensor([[1, 1], [1, 0]]),
            "old_log_probs": torch.zeros((2, 2)),
            "advantages": torch.ones((2, 2)),
            "dapo_reference_teacher_input_ids": torch.tensor([[2, 1, 5, 6], [3, 1, 7, 0]]),
            "dapo_reference_teacher_attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
            "dapo_reference_teacher_position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]]),
            "dapo_reference_kl_loss_mask": torch.tensor([[1, 1], [0, 0]]),
        },
        meta_info={
            "temperature": 1.0,
            "pad_token_id": 0,
            "dapo_reference_kl": True,
            "dapo_reference_kl_coef": 0.1,
            "dapo_reference_kl_temperature": 1.0,
            "dapo_reference_kl_approximation": "topk",
            "dapo_reference_kl_top_k": 3,
            "dapo_reference_kl_token_chunk_size": 1,
        },
    )

    metrics = actor.update_policy(data)

    assert metrics["actor/dapo_reference_kl_tokens"] == [2, 0]
    assert metrics["actor/dapo_reference_kl_coef"] == [0.1, 0.1]
    assert recorded_chunk_sizes == [1]
