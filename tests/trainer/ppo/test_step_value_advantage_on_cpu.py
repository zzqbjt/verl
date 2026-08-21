import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_grpo_outcome_advantage,
    compute_step_value_advantage,
    prepare_step_value_context,
)


def _example_batch(*, auxiliary_rewards=None):
    response_mask = torch.ones(2, 7)
    step_end_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    step_end_mask[:, [1, 3, 5]] = True
    step_values = torch.zeros_like(response_mask)
    step_values[0, [1, 3, 5]] = torch.tensor([0.2, 0.6, 0.9])
    step_values[1, [1, 3, 5]] = torch.tensor([0.8, 0.4, 0.1])
    targets = torch.tensor([1.0, 0.0])
    task_rewards = torch.tensor([1.0, -1.0])
    if auxiliary_rewards is None:
        auxiliary_rewards = torch.zeros(2)
    token_rewards = torch.zeros_like(response_mask)
    token_rewards[:, -1] = task_rewards + auxiliary_rewards
    group_ids = np.array(["prompt", "prompt"], dtype=object)
    return (
        token_rewards,
        response_mask,
        step_end_mask,
        step_values,
        targets,
        task_rewards,
        group_ids,
    )


def _compute(example, *, ready=True, lam=0.9, zero_uniform=False, norm_by_group_std=True):
    (
        token_rewards,
        response_mask,
        step_end_mask,
        step_values,
        targets,
        task_rewards,
        group_ids,
    ) = example
    # The probe provider emits probabilities and anchors the last endpoint to
    # the verified binary outcome before the generic estimator sees it.
    step_values = step_values.clone()
    terminal_positions = (
        torch.arange(step_end_mask.shape[1]).expand_as(step_end_mask).masked_fill(~step_end_mask, -1).max(dim=-1).values
    )
    step_values[torch.arange(step_values.shape[0]), terminal_positions] = targets
    initial_values, scales, active = prepare_step_value_context(
        targets,
        group_ids,
        norm_by_group_std=norm_by_group_std,
        zero_when_group_uniform=zero_uniform,
    )
    return compute_step_value_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=group_ids,
        step_values=step_values,
        step_end_mask=step_end_mask,
        step_value_initial_values=initial_values,
        step_value_scales=scales,
        step_value_active=active,
        step_value_task_rewards=task_rewards,
        step_value_ready=ready,
        lam=lam,
        norm_by_group_std=norm_by_group_std,
    )


def test_step_value_warmup_exactly_matches_grpo_on_total_reward():
    example = _example_batch(auxiliary_rewards=torch.tensor([-0.5, 0.0]))
    token_rewards, response_mask, _, _, _, _, group_ids = example
    expected, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=group_ids,
        norm_adv_by_std_in_grpo=True,
    )

    actual, returns = _compute(example, ready=False)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(returns, expected)


def test_lambda_one_telescopes_from_leave_one_out_mean_and_broadcasts_by_step():
    advantages, _ = _compute(_example_batch(), ready=True, lam=1.0)

    # Each response starts from the other response's acc.  With lambda=1,
    # each trace telescopes from the value immediately before that step to the
    # verified terminal, then uses the acc group's sample std.
    group_scale = torch.tensor([1.0, 0.0]).std(unbiased=True) + 1e-6
    expected_unscaled = torch.tensor(
        [
            [1.0, 1.0, 0.8, 0.8, 0.4, 0.4, 0.4],
            [-1.0, -1.0, -0.8, -0.8, -0.4, -0.4, -0.4],
        ]
    )
    expected = expected_unscaled / group_scale
    torch.testing.assert_close(advantages, expected)


def test_fixed_lambda_keeps_raw_local_value_trace_without_token_centering():
    example = _example_batch()
    advantages, _ = _compute(example, ready=True, lam=0.9)

    group_scale = torch.tensor([1.0, 0.0]).std(unbiased=True) + 1e-6
    positive_deltas = torch.tensor([0.2, 0.4, 0.4]) / group_scale
    expected_steps = torch.tensor(
        [
            positive_deltas[0] + 0.9 * positive_deltas[1] + 0.9**2 * positive_deltas[2],
            positive_deltas[1] + 0.9 * positive_deltas[2],
            positive_deltas[2],
        ]
    )
    expected_positive = expected_steps[[0, 0, 1, 1, 2, 2, 2]]
    expected = torch.stack((expected_positive, -expected_positive))
    torch.testing.assert_close(advantages, expected)

    # A conservation rewrite would force this mean back to the sequence RLOO
    # value (+/-1/std(acc)); the raw local trace does not.
    actual_means = advantages.mean(dim=-1)
    sequence_rloo = torch.tensor([1.0, -1.0]) / group_scale
    assert not torch.allclose(actual_means, sequence_rloo)


def test_one_positive_seven_negative_single_step_uses_rloo_v0_and_group_std():
    batch_size = 8
    response_mask = torch.ones(batch_size, 2)
    step_end_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    step_end_mask[:, 0] = True
    targets = torch.tensor([1.0] + [0.0] * 7)
    task_rewards = torch.tensor([1.0] + [-1.0] * 7)
    token_rewards = torch.zeros_like(response_mask)
    token_rewards[:, -1] = task_rewards

    group_ids = np.array(["prompt"] * batch_size, dtype=object)
    initial_values, scales, active = prepare_step_value_context(
        targets,
        group_ids,
        norm_by_group_std=True,
        zero_when_group_uniform=False,
    )
    step_values = torch.zeros_like(response_mask)
    step_values[:, 0] = targets
    advantages, _ = compute_step_value_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=group_ids,
        step_values=step_values,
        step_end_mask=step_end_mask,
        step_value_initial_values=initial_values,
        step_value_scales=scales,
        step_value_active=active,
        step_value_task_rewards=task_rewards,
        step_value_ready=True,
        lam=0.9,
        norm_by_group_std=True,
    )

    group_scale = targets.std(unbiased=True) + 1e-6
    expected_positive = (targets[0] - targets[1:].mean()) / group_scale
    assert float(expected_positive) == pytest.approx(2.82842, abs=1e-4)
    torch.testing.assert_close(advantages[0], expected_positive.expand(2))

    expected_negative = (targets[1] - targets[[0, 2, 3, 4, 5, 6, 7]].mean()) / group_scale
    torch.testing.assert_close(advantages[1], expected_negative.expand(2))


def test_single_step_rloo_v0_can_disable_group_std_normalization():
    batch_size = 8
    response_mask = torch.ones(batch_size, 1)
    targets = torch.tensor([1.0] + [0.0] * 7)
    task_rewards = torch.tensor([1.0] + [-1.0] * 7)

    group_ids = np.array(["prompt"] * batch_size, dtype=object)
    initial_values, scales, active = prepare_step_value_context(
        targets,
        group_ids,
        norm_by_group_std=False,
        zero_when_group_uniform=False,
    )
    advantages, _ = compute_step_value_advantage(
        token_level_rewards=task_rewards.unsqueeze(-1),
        response_mask=response_mask,
        index=group_ids,
        step_values=targets.unsqueeze(-1),
        step_end_mask=torch.ones_like(response_mask, dtype=torch.bool),
        step_value_initial_values=initial_values,
        step_value_scales=scales,
        step_value_active=active,
        step_value_task_rewards=task_rewards,
        step_value_ready=True,
        lam=0.9,
        norm_by_group_std=False,
    )

    expected = torch.tensor([1.0] + [-1.0 / 7.0] * 7).unsqueeze(-1)
    torch.testing.assert_close(advantages, expected)


def test_generic_estimator_accepts_provider_prepared_raw_reward_values():
    final_rewards = torch.tensor([-2.0, 3.0])
    group_ids = np.array(["prompt", "prompt"], dtype=object)
    initial_values, scales, active = prepare_step_value_context(
        final_rewards,
        group_ids,
        norm_by_group_std=True,
        zero_when_group_uniform=True,
    )
    step_values = torch.tensor([[0.0, -2.0], [1.0, 3.0]])
    token_rewards = torch.zeros_like(step_values)
    token_rewards[:, -1] = final_rewards

    advantages, _ = compute_step_value_advantage(
        token_level_rewards=token_rewards,
        response_mask=torch.ones_like(step_values),
        index=group_ids,
        step_values=step_values,
        step_end_mask=torch.ones_like(step_values, dtype=torch.bool),
        step_value_initial_values=initial_values,
        step_value_scales=scales,
        step_value_active=active,
        step_value_task_rewards=final_rewards,
        step_value_ready=True,
        lam=1.0,
        norm_by_group_std=True,
    )

    expected = torch.tensor([[-5.0, -2.0], [5.0, 2.0]]) / scales[0]
    torch.testing.assert_close(advantages, expected)


def test_uniform_outcome_group_cannot_create_probe_only_advantage_when_disabled():
    example = list(_example_batch())
    example[4] = torch.ones(2)
    example[5] = torch.ones(2)
    example[0].zero_()
    example[0][:, -1] = 1.0

    advantages, _ = _compute(tuple(example), ready=True, zero_uniform=True)

    torch.testing.assert_close(advantages, torch.zeros_like(advantages))


def test_uniform_outcome_group_can_use_local_probe_values_when_enabled():
    example = list(_example_batch())
    example[4] = torch.ones(2)
    example[5] = torch.ones(2)
    example[0].zero_()
    example[0][:, -1] = 1.0

    advantages, _ = _compute(tuple(example), ready=True, lam=0.9, zero_uniform=False)

    expected_step_advantages = torch.tensor(
        [
            [-0.8 + 0.9 * 0.4 + 0.9**2 * 0.4, 0.4 + 0.9 * 0.4, 0.4],
            [-0.2 + 0.9 * -0.4 + 0.9**2 * 0.6, -0.4 + 0.9 * 0.6, 0.6],
        ]
    )
    expected = expected_step_advantages[:, [0, 0, 1, 1, 2, 2, 2]]
    torch.testing.assert_close(advantages, expected)


def test_auxiliary_reward_centers_on_full_group_and_uses_task_group_std():
    auxiliary_rewards = torch.tensor([-0.5, 0.0])
    with_auxiliary = _example_batch(auxiliary_rewards=auxiliary_rewards)
    without_auxiliary = _example_batch()

    actual, _ = _compute(with_auxiliary, ready=True, lam=0.9)
    baseline, _ = _compute(without_auxiliary, ready=True, lam=0.9)

    centered_auxiliary = auxiliary_rewards - auxiliary_rewards.mean()
    task_group_scale = with_auxiliary[5].std(unbiased=True) + 1e-6
    expected_scalar = centered_auxiliary / task_group_scale
    expected = expected_scalar.unsqueeze(-1).expand_as(actual)
    torch.testing.assert_close(actual - baseline, expected)

    # Leave-one-out centering would give twice this magnitude for a two-item
    # group.  Auxiliary rewards deliberately retain full-group centering while
    # using the same task-reward std as the task step deltas.
    rloo_magnitude = 0.5 / task_group_scale
    assert abs(float((actual - baseline)[0, 0])) < rloo_magnitude


def test_probe_provider_anchors_terminal_prediction_before_generic_advantage():
    example = list(_example_batch())
    first, _ = _compute(tuple(example), ready=True)
    changed_values = example[3].clone()
    changed_values[0, 5] = 0.0
    changed_values[1, 5] = 1.0
    example[3] = changed_values

    second, _ = _compute(tuple(example), ready=True)

    torch.testing.assert_close(first, second)
