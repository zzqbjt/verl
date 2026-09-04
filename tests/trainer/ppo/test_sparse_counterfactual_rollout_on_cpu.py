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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
from recipe.dapo.sparse_counterfactual_credit import SparseCounterfactualCreditSupervisor
from verl import DataProto
from verl.trainer.config import SparseCounterfactualCreditConfig
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import compute_advantage


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
    )
    supervisor.rollout_config = OmegaConf.create({"max_model_len": 100})
    seen_prompts = []
    seen_full_responses = []

    def generate(prompt_ids, response_prefix_lengths):
        # Initial anchors skip V rollouts; terminal anchors skip Q rollouts.
        assert response_prefix_lengths == [2, 2, 2]
        seen_prompts.extend(prompt_ids)
        return [SimpleNamespace(token_ids=[90 + index]) for index in range(len(prompt_ids))]

    def score(batch, source_indices, response_ids):
        del batch, source_indices
        seen_full_responses.extend(response_ids)
        return torch.tensor([0.5, 0.25, 0.25])

    supervisor._generate = generate
    supervisor._score = score
    batch = _make_batch()

    forced_anchor_mask = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.bool)
    with patch(
        "recipe.dapo.sparse_counterfactual_credit.sample_anchor_steps",
        return_value=forced_anchor_mask,
    ):
        result = supervisor.collect_targets(batch, global_step=1)
        metrics = result.metrics

    assert seen_prompts == [
        [1, 2, 10, 11],
        [1, 2, 20, 21],
        [1, 2, 20, 21],
    ]
    assert seen_full_responses[0] == [10, 11, 90]
    assert seen_full_responses[1] == [20, 21, 91]
    anchor_targets = batch.batch["credit_anchor_targets"][batch.batch["credit_anchor_mask"]]
    # First-step V uses the full group's correctness.
    # Terminal-step Q uses its own final correctness without a Q rollout.
    torch.testing.assert_close(anchor_targets, torch.tensor([0.0, -0.25]))
    torch.testing.assert_close(
        batch.batch["credit_anchor_q_targets"][batch.batch["credit_anchor_mask"]],
        torch.tensor([0.5, 0.0]),
    )
    torch.testing.assert_close(
        batch.batch["credit_anchor_v_targets"][batch.batch["credit_anchor_mask"]],
        torch.tensor([0.5, 0.25]),
    )
    torch.testing.assert_close(batch.batch["credit_start_value_targets"], torch.tensor([0.5, 0.5]))
    assert torch.equal(batch.batch["credit_start_value_train_mask"], torch.tensor([True, False]))
    torch.testing.assert_close(batch.batch["credit_terminal_value_targets"], torch.tensor([1.0, 0.0]))
    assert "credit_anchor_weights" not in batch.batch
    assert "credit_anchor_probabilities" not in batch.batch
    assert metrics["credit/q_mean"] == 0.25
    assert metrics["credit/v_mean"] == 0.375
    assert "credit/anchors" not in metrics
    assert "credit/zero_budget_suffixes" not in metrics
    assert result.training_branches is None


def test_middle_steps_keep_both_q_and_v_suffix_rollouts():
    supervisor = SparseCounterfactualCreditSupervisor.__new__(SparseCounterfactualCreditSupervisor)
    supervisor.config = SparseCounterfactualCreditConfig(
        enabled=True,
        anchors_per_group=6,
        num_q_samples=1,
        num_v_samples=2,
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

    batch = _make_three_step_batch()
    with patch(
        "recipe.dapo.sparse_counterfactual_credit.sample_anchor_steps",
        return_value=batch.batch["step_end_mask"].clone(),
    ):
        result = supervisor.collect_targets(batch, global_step=1)

    # Per response: first Q; middle Q + two V; final two V.
    assert seen_prefix_lengths == [2, 4, 2, 2, 4, 4] * 2
    assert result.training_branches is None


def test_single_step_responses_need_no_counterfactual_suffix_rollouts():
    supervisor = SparseCounterfactualCreditSupervisor.__new__(SparseCounterfactualCreditSupervisor)
    supervisor.config = SparseCounterfactualCreditConfig(
        enabled=True,
        anchors_per_group=2,
        num_q_samples=1,
        num_v_samples=2,
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

    result = supervisor.collect_targets(batch, global_step=1)
    metrics = result.metrics

    anchor_targets = batch.batch["credit_anchor_targets"][batch.batch["credit_anchor_mask"]]
    torch.testing.assert_close(anchor_targets, torch.tensor([0.5, -0.5]))
    assert metrics["credit/q_mean"] == 0.5
    assert metrics["credit/v_mean"] == 0.5


def test_mc_branch_anchor_candidates_exclude_single_step_responses_and_keep_late_steps():
    supervisor = SparseCounterfactualCreditSupervisor.__new__(SparseCounterfactualCreditSupervisor)
    supervisor.config = SparseCounterfactualCreditConfig(
        enabled=True,
        anchors_per_group=3,
        train_mc_branches=True,
    )
    supervisor.trainer_config = OmegaConf.create(
        {
            "data": {"max_response_length": 3},
            "reward": {"reward_kwargs": {"overlong_buffer_cfg": {"len": 1}}},
        }
    )
    step_end_mask = torch.tensor(
        [
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
        ],
        dtype=torch.bool,
    )
    batch = DataProto.from_dict(
        tensors={"step_end_mask": step_end_mask, "response_mask": torch.ones_like(step_end_mask)},
        non_tensors={"uid": np.asarray(["enough"] * 4 + ["short"] * 4, dtype=object)},
    )

    candidates = supervisor._mc_branch_anchor_candidates(step_end_mask, batch.batch["response_mask"])
    assert not candidates[[1, 5, 7]].any()
    assert candidates[[0, 2, 3, 4, 6]].any(dim=-1).all()
    assert candidates[[2, 3, 6], 2].all()
    assert supervisor.mc_branch_eligible_prompt_uids(batch) == {"enough"}


def test_mc_branch_training_selects_diverse_high_variance_anchors_and_masks_prefixes():
    supervisor = SparseCounterfactualCreditSupervisor.__new__(SparseCounterfactualCreditSupervisor)
    supervisor.config = SparseCounterfactualCreditConfig(
        enabled=True,
        anchors_per_group=2,
        num_q_samples=2,
        num_v_samples=2,
        train_mc_branches=True,
        branch_groups_per_prompt=2,
        selection_seed=7,
    )
    supervisor.rollout_config = OmegaConf.create({"max_model_len": 100})
    supervisor.trainer_config = OmegaConf.create({"data": {"max_response_length": 6}})
    supervisor.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=99)
    batch = _make_three_step_batch()
    scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    scores[:, -1] = torch.tensor([1.0, 0.0])
    batch.batch["token_level_scores"] = scores
    batch.batch["token_level_rewards"] = scores.clone()

    def generate(prompt_ids, response_prefix_lengths):
        assert len(prompt_ids) == 8
        return [SimpleNamespace(token_ids=[90 + index]) for index in range(8)]

    def score(batch, source_indices, response_ids):
        del batch, source_indices, response_ids
        # row-0 Q/V and row-1 Q all have maximum binary variance.  The
        # higher-entropy row-1 anchor wins first, then anchor diversity wins
        # over selecting another group from row 1.  Give row-1 V a much larger
        # shaped-reward variance to ensure selection only uses correctness.
        return SimpleNamespace(
            correctness=torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
            rewards=torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, -10.0, 10.0]),
        )

    supervisor._generate = generate
    supervisor._score = score
    forced_anchor_mask = torch.tensor(
        [[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]],
        dtype=torch.bool,
    )
    with patch(
        "recipe.dapo.sparse_counterfactual_credit.sample_anchor_steps",
        return_value=forced_anchor_mask,
    ):
        result = supervisor.collect_targets(batch, global_step=3)

    branches = result.training_branches
    assert branches is not None
    assert len(branches) == 4
    assert torch.equal(branches.batch["response_mask"].sum(dim=-1), torch.ones(4, dtype=torch.long))
    for row, prefix_length in enumerate(branches.batch["credit_prefix_lengths"].tolist()):
        assert not branches.batch["response_mask"][row, :prefix_length].any()
        assert branches.batch["response_mask"][row, prefix_length]
    group_uids = branches.non_tensor_batch["uid"].tolist()
    assert len(set(group_uids)) == 2
    assert all(group_uids.count(uid) == 2 for uid in set(group_uids))
    selected_anchors = {int(uid.rsplit(":", 2)[1]) for uid in group_uids}
    assert selected_anchors == {0, 1}
    assert any(uid.endswith(":1:q") for uid in group_uids)
    assert not any(uid.endswith(":1:v") for uid in group_uids)
    assert branches.batch["is_mc_branch"].all()
    assert branches.batch["credit_start_value_mask"].all()
    assert not branches.batch["credit_start_value_train_mask"].any()
    assert branches.batch["credit_terminal_value_mask"].all()
    torch.testing.assert_close(
        branches.batch["credit_terminal_value_targets"],
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )


def test_dapo_trainer_appends_branch_schema_without_unmasking_fixed_prefixes():
    base = _make_three_step_batch()
    scores = torch.zeros_like(base.batch["responses"], dtype=torch.float32)
    scores[:, -1] = torch.tensor([1.0, 0.0])
    base.batch["token_level_scores"] = scores
    base.batch["token_level_rewards"] = scores.clone()
    base.batch["old_log_probs"] = torch.zeros_like(scores)
    base.batch["credit_anchor_mask"] = torch.zeros_like(scores, dtype=torch.bool)
    base.batch["credit_anchor_mask"][:, 3] = True
    base.batch["credit_anchor_targets"] = torch.zeros_like(scores)
    base.batch["credit_anchor_q_targets"] = torch.zeros_like(scores)
    base.batch["credit_anchor_v_targets"] = torch.zeros_like(scores)
    base.batch["credit_start_value_mask"] = torch.ones(2, dtype=torch.bool)
    base.batch["credit_start_value_train_mask"] = torch.tensor([True, False])
    base.batch["credit_start_value_targets"] = torch.full((2,), 0.5)
    base.batch["credit_terminal_value_mask"] = torch.ones(2, dtype=torch.bool)
    base.batch["credit_terminal_value_targets"] = torch.tensor([1.0, 0.0])
    base.meta_info["global_token_num"] = base.batch["attention_mask"].sum(dim=-1).tolist()

    branches = base.select_idxs([0, 0, 1, 1])
    branches.batch["response_mask"] = torch.zeros_like(branches.batch["response_mask"])
    branches.batch["response_mask"][:, 2:5] = True
    branches.batch["credit_prefix_lengths"] = torch.full((4,), 2, dtype=torch.long)
    branches.batch["is_mc_branch"] = torch.ones(4, dtype=torch.bool)
    branches.batch["credit_start_value_train_mask"] = torch.zeros(4, dtype=torch.bool)
    branches.batch["credit_start_value_targets"] = torch.tensor([0.75, 0.75, 0.25, 0.25])
    branches.batch["credit_terminal_value_targets"] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    for key in (
        "step_end_mask",
        "entropys",
        "old_log_probs",
        "credit_anchor_mask",
        "credit_anchor_targets",
        "credit_anchor_q_targets",
        "credit_anchor_v_targets",
    ):
        branches.batch.pop(key)
    branches.non_tensor_batch["uid"] = np.asarray(["q", "q", "v", "v"], dtype=object)

    trainer = SimpleNamespace(use_reference_policy=False)

    def prepare_steps(data):
        data.batch["step_end_mask"] = torch.zeros_like(data.batch["response_mask"], dtype=torch.bool)
        data.batch["step_end_mask"][:, 4] = True

    def compute_old_log_prob(data):
        shape = data.batch["responses"].shape
        return (
            DataProto.from_dict(
                tensors={
                    "old_log_probs": torch.zeros(shape),
                    "entropys": torch.ones(shape),
                }
            ),
            0.0,
        )

    trainer._prepare_step_inputs = prepare_steps
    trainer._compute_old_log_prob = compute_old_log_prob
    with patch("recipe.dapo.dapo_ray_trainer.marked_timer", return_value=nullcontext()):
        combined = RayDAPOTrainer._append_mc_training_branches(trainer, base, branches, {})

    assert len(combined) == 6
    assert not combined.batch["is_mc_branch"][:2].any()
    assert combined.batch["is_mc_branch"][2:].all()
    assert torch.equal(combined.batch["credit_prefix_lengths"], torch.tensor([0, 0, 2, 2, 2, 2]))
    assert not combined.batch["response_mask"][2:, :2].any()
    assert combined.batch["response_mask"][2:, 2:5].all()
    torch.testing.assert_close(
        combined.batch["credit_start_value_targets"],
        torch.tensor([0.5, 0.5, 0.75, 0.75, 0.25, 0.25]),
    )

    combined.batch["token_level_rewards"].zero_()
    combined.batch["token_level_rewards"][:, -1] = torch.tensor([0.0, 1.0, 10.0, 0.0, 5.0, 5.0])
    combined = compute_advantage(
        combined,
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=True,
    )
    scale = torch.tensor(2.0**-0.5)
    torch.testing.assert_close(combined.batch["advantages"][:2, 0], torch.tensor([-scale, scale]))
    torch.testing.assert_close(combined.batch["advantages"][2:4, 2], torch.tensor([scale, -scale]))
    torch.testing.assert_close(combined.batch["advantages"][4:, 2], torch.zeros(2))
    assert not combined.batch["advantages"][2:, :2].any()


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


def test_counterfactual_suffix_budget_uses_full_response_length_despite_overlong_buffer():
    supervisor = SparseCounterfactualCreditSupervisor.__new__(SparseCounterfactualCreditSupervisor)
    supervisor.config = SparseCounterfactualCreditConfig(max_new_tokens=None)
    supervisor.trainer_config = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 20,
                "max_response_length": 10,
            },
            "reward": {
                "reward_kwargs": {
                    "overlong_buffer_cfg": {
                        "enable": True,
                        "len": 2,
                    }
                }
            },
        }
    )
    supervisor.rollout_config = OmegaConf.create({"max_model_len": 30})

    assert supervisor._counterfactual_response_limit() == 10
    assert supervisor._sampling_params(prompt_length=15, response_prefix_length=3)["max_tokens"] == 7
    assert supervisor._sampling_params(prompt_length=28, response_prefix_length=8)["max_tokens"] == 2
