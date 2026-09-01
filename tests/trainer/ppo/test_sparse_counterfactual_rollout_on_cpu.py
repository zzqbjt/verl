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

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from recipe.dapo.sparse_counterfactual_credit import SparseCounterfactualCreditSupervisor
from verl import DataProto
from verl.trainer.config import SparseCounterfactualCreditConfig


def _make_batch() -> DataProto:
    prompts = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    responses = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long)
    attention_mask = torch.ones(2, 7, dtype=torch.long)
    attention_mask[:, 0] = 0
    response_mask = torch.ones(2, 4, dtype=torch.bool)
    step_end_mask = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=torch.bool)
    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": attention_mask,
            "position_ids": attention_mask.cumsum(dim=-1) - 1,
            "response_mask": response_mask,
            "step_end_mask": step_end_mask,
            "entropys": torch.tensor([[1.0, 3.0, 2.0, 4.0], [5.0, 1.0, 2.0, 6.0]]),
        },
        batch_size=[2],
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "uid": np.asarray(["same-question", "same-question"], dtype=object),
            "acc": np.asarray([1.0, 0.0], dtype=np.float32),
        },
    )


def _make_three_step_batch() -> DataProto:
    prompts = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    responses = torch.tensor(
        [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]],
        dtype=torch.long,
    )
    attention_mask = torch.ones(2, 9, dtype=torch.long)
    attention_mask[:, 0] = 0
    response_mask = torch.ones(2, 6, dtype=torch.bool)
    step_end_mask = torch.tensor(
        [[0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]],
        dtype=torch.bool,
    )
    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": attention_mask,
            "position_ids": attention_mask.cumsum(dim=-1) - 1,
            "response_mask": response_mask,
            "step_end_mask": step_end_mask,
            "entropys": torch.arange(12, dtype=torch.float32).reshape(2, 6),
        },
        batch_size=[2],
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "uid": np.asarray(["same-question", "same-question"], dtype=object),
            "acc": np.asarray([1.0, 0.0], dtype=np.float32),
        },
    )


def test_counterfactual_rollout_reuses_boundary_values_and_samples_only_required_suffixes():
    supervisor = SparseCounterfactualCreditSupervisor.__new__(SparseCounterfactualCreditSupervisor)
    supervisor.config = SparseCounterfactualCreditConfig(
        enabled=True,
        anchors_per_group=4,
        num_q_samples=1,
        num_v_samples=2,
        use_inverse_propensity=False,
    )
    supervisor.rollout_config = OmegaConf.create({"max_model_len": 100})
    seen_prompts = []
    seen_full_responses = []

    def generate(prompt_ids, response_prefix_lengths):
        # Initial anchors skip V rollouts; terminal anchors skip Q rollouts.
        assert response_prefix_lengths == [2, 2, 2, 2, 2, 2]
        seen_prompts.extend(prompt_ids)
        return [SimpleNamespace(token_ids=[90 + index]) for index in range(len(prompt_ids))]

    def score(batch, source_indices, response_ids):
        del batch, source_indices
        seen_full_responses.extend(response_ids)
        return torch.tensor([0.5, 0.25, 0.25, 0.5, 0.25, 0.25])

    supervisor._generate = generate
    supervisor._score = score
    batch = _make_batch()

    metrics = supervisor.collect_targets(batch, global_step=1)

    assert seen_prompts == [
        [1, 2, 10, 11],
        [1, 2, 10, 11],
        [1, 2, 10, 11],
        [1, 2, 20, 21],
        [1, 2, 20, 21],
        [1, 2, 20, 21],
    ]
    assert seen_full_responses[0] == [10, 11, 90]
    assert seen_full_responses[1] == [10, 11, 91]
    assert seen_full_responses[3] == [20, 21, 93]
    anchor_targets = batch.batch["credit_anchor_targets"][batch.batch["credit_anchor_mask"]]
    # First-step V uses the other response's correctness (leave one out).
    # Terminal-step Q uses its own final correctness without a Q rollout.
    torch.testing.assert_close(anchor_targets, torch.tensor([0.75, 0.75, -0.75, -0.25]))
    assert torch.equal(batch.batch["credit_anchor_weights"][batch.batch["credit_anchor_mask"]], torch.ones(4))
    assert metrics["credit/q_mean"] == 0.5
    assert metrics["credit/v_mean"] == 0.375
    assert "credit/anchors" not in metrics
    assert "credit/zero_budget_suffixes" not in metrics


def test_middle_steps_keep_both_q_and_v_suffix_rollouts():
    supervisor = SparseCounterfactualCreditSupervisor.__new__(SparseCounterfactualCreditSupervisor)
    supervisor.config = SparseCounterfactualCreditConfig(
        enabled=True,
        anchors_per_group=6,
        num_q_samples=1,
        num_v_samples=2,
        use_inverse_propensity=False,
    )
    supervisor.rollout_config = OmegaConf.create({"max_model_len": 100})
    seen_prefix_lengths = []

    def generate(prompt_ids, response_prefix_lengths):
        assert len(prompt_ids) == 12
        seen_prefix_lengths.extend(response_prefix_lengths)
        return [SimpleNamespace(token_ids=[90]) for _ in prompt_ids]

    def score(batch, source_indices, response_ids):
        del batch, source_indices
        return torch.full((len(response_ids),), 0.25)

    supervisor._generate = generate
    supervisor._score = score

    supervisor.collect_targets(_make_three_step_batch(), global_step=1)

    # Per response: first Q; middle Q + two V; final two V.
    assert seen_prefix_lengths == [2, 4, 2, 2, 4, 4] * 2


def test_single_step_responses_need_no_counterfactual_suffix_rollouts():
    supervisor = SparseCounterfactualCreditSupervisor.__new__(SparseCounterfactualCreditSupervisor)
    supervisor.config = SparseCounterfactualCreditConfig(
        enabled=True,
        anchors_per_group=2,
        num_q_samples=1,
        num_v_samples=2,
        use_inverse_propensity=False,
    )
    supervisor.rollout_config = OmegaConf.create({"max_model_len": 100})

    def fail_generate(*args, **kwargs):
        raise AssertionError(f"single-step responses must not generate suffixes: {args}, {kwargs}")

    def fail_score(*args, **kwargs):
        raise AssertionError(f"single-step responses must not be rescored: {args}, {kwargs}")

    supervisor._generate = fail_generate
    supervisor._score = fail_score
    batch = _make_batch()
    batch.batch["step_end_mask"] = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=torch.bool)

    metrics = supervisor.collect_targets(batch, global_step=1)

    anchor_targets = batch.batch["credit_anchor_targets"][batch.batch["credit_anchor_mask"]]
    torch.testing.assert_close(anchor_targets, torch.tensor([1.0, -1.0]))
    assert metrics["credit/q_mean"] == 0.5
    assert metrics["credit/v_mean"] == 0.5


def test_counterfactual_suffix_budget_tracks_max_response_length_and_context_limit():
    supervisor = SparseCounterfactualCreditSupervisor.__new__(SparseCounterfactualCreditSupervisor)
    supervisor.config = SparseCounterfactualCreditConfig(max_new_tokens=None)
    supervisor.trainer_config = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 20,
                "max_response_length": 10,
            }
        }
    )
    supervisor.rollout_config = OmegaConf.create({"max_model_len": 30})

    assert supervisor._sampling_params(prompt_length=15, response_prefix_length=3)["max_tokens"] == 7
    assert supervisor._sampling_params(prompt_length=28, response_prefix_length=3)["max_tokens"] == 2
    assert supervisor._sampling_params(prompt_length=30, response_prefix_length=10)["max_tokens"] == 0

    class FailIfCalledServerManager:
        async def generate(self, **kwargs):
            raise AssertionError(f"zero-budget suffix must not call vLLM: {kwargs}")

    supervisor.direct_server_manager = FailIfCalledServerManager()
    assert supervisor._generate([list(range(30))], [10]) == [None]


def test_counterfactual_suffix_budget_supports_an_optional_stricter_cap():
    supervisor = SparseCounterfactualCreditSupervisor.__new__(SparseCounterfactualCreditSupervisor)
    supervisor.config = SparseCounterfactualCreditConfig(max_new_tokens=4)
    supervisor.trainer_config = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 20,
                "max_response_length": 10,
            }
        }
    )
    supervisor.rollout_config = OmegaConf.create({"max_model_len": 30})

    assert supervisor._sampling_params(prompt_length=15, response_prefix_length=3)["max_tokens"] == 4
