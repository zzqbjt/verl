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

import torch

from verl.trainer.ppo.sparse_counterfactual_credit import (
    build_credit_residual,
    compute_anchor_probabilities,
    compute_monte_carlo_credit,
    compute_step_uncertainty,
    credit_advantage_coefficient,
    expand_step_credit_to_tokens,
    merge_anchor_credit,
    sample_anchor_steps,
)


def _step_masks():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    step_end_mask = torch.tensor(
        [
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    return step_end_mask, response_mask


def test_top_entropy_uncertainty_uses_ceil_twenty_percent_within_each_step():
    step_end_mask, response_mask = _step_masks()
    entropy = torch.tensor(
        [
            [1.0, 5.0, 2.0, 8.0, 3.0, 99.0],
            [4.0, 1.0, 7.0, 2.0, 99.0, 99.0],
            [6.0, 3.0, 9.0, 1.0, 100.0, 99.0],
        ]
    )

    uncertainty = compute_step_uncertainty(
        entropy,
        step_end_mask,
        response_mask,
        top_ratio=0.2,
    )

    expected = torch.zeros_like(entropy)
    expected[0, 1] = 5.0
    expected[0, 4] = 8.0
    expected[1, 3] = 7.0
    expected[2, 0] = 6.0
    expected[2, 3] = 9.0
    torch.testing.assert_close(uncertainty, expected)


def test_group_sampling_distribution_and_without_replacement_are_deterministic():
    step_end_mask, _ = _step_masks()
    uncertainty = torch.zeros(3, 6)
    uncertainty[step_end_mask] = torch.tensor([0.0, 1.0, 2.0, 0.5, 1.5])
    uids = ["question-a", "question-a", "question-b"]

    probabilities = compute_anchor_probabilities(
        uncertainty,
        step_end_mask,
        uids,
        temperature=1.0,
        uniform_mix=0.1,
    )

    torch.testing.assert_close(probabilities[:2].sum(), torch.tensor(1.0))
    torch.testing.assert_close(probabilities[2].sum(), torch.tensor(1.0))
    assert torch.all(probabilities[step_end_mask] > 0)
    first = sample_anchor_steps(
        probabilities,
        step_end_mask,
        uids,
        anchors_per_group=2,
        seed=7,
        global_step=3,
    )
    second = sample_anchor_steps(
        probabilities,
        step_end_mask,
        uids,
        anchors_per_group=2,
        seed=7,
        global_step=3,
    )
    assert torch.equal(first, second)
    assert int(first[:2].sum()) == 2
    assert int(first[2].sum()) == 2


def test_monte_carlo_anchor_override_and_policy_token_broadcast():
    q_rewards = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    v_rewards = torch.tensor([[0.0, 0.5], [0.5, 0.5]])
    credit = compute_monte_carlo_credit(q_rewards, v_rewards)
    torch.testing.assert_close(credit, torch.tensor([0.25, 0.5]))

    predictions = torch.tensor([[0.0, 0.1, 0.0], [0.2, 0.0, 0.3]])
    anchor_mask = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.bool)
    merged = merge_anchor_credit(predictions, anchor_mask, credit)
    torch.testing.assert_close(merged, torch.tensor([[0.0, 0.25, 0.0], [0.5, 0.0, 0.3]]))

    step_end_mask = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.bool)
    response_mask = torch.ones_like(step_end_mask)
    endpoint_credit = torch.tensor([[0.0, 2.0, 0.0, 4.0, 0.0]])
    expanded = expand_step_credit_to_tokens(endpoint_credit, step_end_mask, response_mask)
    torch.testing.assert_close(expanded, torch.tensor([[2.0, 2.0, 4.0, 4.0, 4.0]]))


def test_credit_residual_preserves_each_response_advantage_sum_and_normalizes_population_std():
    step_end_mask = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    response_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    endpoint_credit = torch.tensor([[0.0, 1.0, 0.0, 3.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0]])

    residual, metrics = build_credit_residual(
        endpoint_credit,
        step_end_mask,
        response_mask,
        normalize_batch_std=True,
        epsilon=1e-12,
    )

    torch.testing.assert_close((residual * response_mask).sum(dim=-1), torch.zeros(2), atol=1e-6, rtol=0)
    population_second_moment = residual[response_mask].square().mean()
    torch.testing.assert_close(population_second_moment, torch.tensor(1.0), atol=1e-6, rtol=0)
    assert set(metrics) == {"credit/residual_abs_mean", "credit/residual_scale"}


def test_credit_advantage_coefficient_starts_at_zero_and_reaches_maximum():
    assert (
        credit_advantage_coefficient(
            global_step=1,
            total_training_steps=100,
            maximum=0.3,
            warmup_ratio=0.1,
        )
        == 0.0
    )
    assert (
        credit_advantage_coefficient(
            global_step=11,
            total_training_steps=100,
            maximum=0.3,
            warmup_ratio=0.1,
        )
        == 0.3
    )
    assert (
        credit_advantage_coefficient(
            global_step=1,
            total_training_steps=100,
            maximum=0.3,
            warmup_ratio=0.0,
        )
        == 0.3
    )
