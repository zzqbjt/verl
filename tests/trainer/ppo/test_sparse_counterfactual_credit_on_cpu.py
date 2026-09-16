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

import pytest
import torch

from verl.trainer.ppo.sparse_counterfactual_credit import (
    build_credit_residual,
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


def test_group_sampling_is_deterministic_and_uses_distinct_responses():
    step_end_mask, _ = _step_masks()
    uncertainty = torch.zeros(3, 6)
    uncertainty[step_end_mask] = torch.tensor([0.0, 1.0, 2.0, 0.5, 1.5])
    uids = ["question-a", "question-a", "question-a"]

    first = sample_anchor_steps(
        uncertainty,
        step_end_mask,
        uids,
        temperature=1.0,
        uniform_mix=0.1,
        anchors_per_group=2,
        seed=7,
        global_step=3,
    )
    second = sample_anchor_steps(
        uncertainty,
        step_end_mask,
        uids,
        temperature=1.0,
        uniform_mix=0.1,
        anchors_per_group=2,
        seed=7,
        global_step=3,
    )
    assert torch.equal(first, second)
    assert int(first.sum()) == 2
    assert torch.all(first.sum(dim=-1) <= 1)


def test_group_sampling_rejects_more_anchors_than_responses():
    step_end_mask, _ = _step_masks()
    uncertainty = step_end_mask.float()

    with pytest.raises(ValueError, match="responses with step endpoints"):
        sample_anchor_steps(
            uncertainty,
            step_end_mask,
            ["question-a"] * 3,
            anchors_per_group=4,
        )


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


def test_credit_residual_preserves_each_response_advantage_sum_and_normalizes_batch_rms():
    step_end_mask = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    response_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    endpoint_credit = torch.tensor([[0.0, 1.0, 0.0, 3.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0]])

    residual, metrics = build_credit_residual(
        endpoint_credit,
        step_end_mask,
        response_mask,
        uids=["same-group", "same-group"],
        epsilon=1e-12,
    )

    torch.testing.assert_close((residual * response_mask).sum(dim=-1), torch.zeros(2), atol=1e-6, rtol=0)
    population_second_moment = residual[response_mask].square().mean()
    torch.testing.assert_close(population_second_moment, torch.tensor(1.0), atol=1e-6, rtol=0)
    assert set(metrics) == {"credit/residual_abs_mean", "credit/residual_scale"}


def _credit_group_batch():
    # Interleave prompt and MC prefix groups, with unequal lengths and row RMS.
    specifications = [
        ("prompt-a", 0, 4, 2, 0.1, 0.5),
        ("prompt-a:mc-branch:7:0:q", 2, 4, 2, 0.1, 0.3),
        ("prompt-b", 0, 7, 1, -0.8, 0.6),
        ("prompt-a", 0, 7, 3, -0.4, 0.3),
        ("prompt-a:mc-branch:7:1:v", 1, 4, 2, -0.5, 0.5),
        ("prompt-b", 0, 3, 3, 0.4, 0.4),
        ("prompt-a:mc-branch:7:0:q", 3, 4, 2, -0.1, 0.5),
        ("prompt-a:mc-branch:7:1:v", 2, 3, 1, -0.1, 0.2),
    ]
    # Non-endpoint values must never enter the normalization.
    credits = torch.full((len(specifications), 8), 999.0, dtype=torch.float64)
    ends = torch.zeros_like(credits, dtype=torch.bool)
    active = torch.zeros_like(ends)
    uids = []
    for row, (uid, prefix, length, first_length, first_credit, last_credit) in enumerate(specifications):
        uids.append(uid)
        active[row, prefix : prefix + length] = True
        first_end, last_end = prefix + first_length - 1, prefix + length - 1
        ends[row, first_end] = ends[row, last_end] = True
        credits[row, first_end] = first_credit
        credits[row, last_end] = last_credit
    return credits, ends, active, uids


def test_other_prompts_and_mc_groups_contribute_to_the_shared_batch_scale():
    credits, ends, active, uids = _credit_group_batch()
    reference, _ = build_credit_residual(credits, ends, active, uids)
    changed = credits.clone()
    changed[2] *= 10
    actual, _ = build_credit_residual(changed, ends, active, uids)
    assert not torch.allclose(actual[0], reference[0])
    assert actual[0].abs().max() < reference[0].abs().max()


def test_credit_batch_scaling_is_invariant_to_batch_reordering():
    credits, ends, active, uids = _credit_group_batch()
    reference, reference_metrics = build_credit_residual(credits, ends, active, uids)
    permutation = torch.tensor([7, 2, 0, 6, 1, 5, 3, 4])
    actual, actual_metrics = build_credit_residual(
        credits[permutation],
        ends[permutation],
        active[permutation],
        [uids[row] for row in permutation.tolist()],
    )
    torch.testing.assert_close(actual, reference[permutation])
    assert actual_metrics == pytest.approx(reference_metrics)


def test_credit_batch_scaling_handles_zero_variance_and_singleton_groups():
    credits = torch.tensor([[0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5]])
    active = torch.ones_like(credits, dtype=torch.bool)
    residual, metrics = build_credit_residual(credits, active, active, ["zero", "zero", "singleton"])
    assert torch.isfinite(residual).all()
    assert not residual[:2].any()
    torch.testing.assert_close(residual[2], torch.tensor([-3.0**0.5, 3.0**0.5]), atol=1e-5, rtol=0)
    assert metrics["credit/residual_scale"] == pytest.approx((0.5 / 6)**0.5)


def test_batch_rms_restores_token_weighted_full_batch_formula():
    credits, ends, active, uids = _credit_group_batch()
    expanded = expand_step_credit_to_tokens(credits, ends, active)
    centered = (expanded - (expanded.sum(-1) / active.sum(-1)).unsqueeze(-1)) * active
    scale = centered[active].square().mean().sqrt()
    actual, metrics = build_credit_residual(
        credits, ends, active, uids
    )
    torch.testing.assert_close(actual, centered / (scale + 1e-6))
    assert metrics["credit/residual_scale"] == pytest.approx(scale.item())
    assert not actual[~active].any()
    torch.testing.assert_close(actual.sum(-1), torch.zeros_like(actual.sum(-1)), atol=1e-5, rtol=0)
    different_uids, _ = build_credit_residual(
        credits, ends, active, ["all"] * len(uids)
    )
    torch.testing.assert_close(actual, different_uids)


def test_batch_rms_zero_variance():
    credits = torch.ones(2, 3)
    active = torch.ones_like(credits, dtype=torch.bool)
    residual, metrics = build_credit_residual(
        credits, active, active, ["a", "b"]
    )
    assert torch.isfinite(residual).all() and not residual.any()
    assert metrics["credit/residual_scale"] == 0


def test_credit_rejects_uid_count_mismatch():
    credits, ends, active, uids = _credit_group_batch()
    with pytest.raises(ValueError, match="uids length"):
        build_credit_residual(credits, ends, active, uids[:-1])


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
