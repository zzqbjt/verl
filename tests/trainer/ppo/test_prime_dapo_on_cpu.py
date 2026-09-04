import numpy as np
import pytest
import torch

from recipe.dapo.prime_dapo_ray_trainer import compute_prime_process_returns


def test_prime_process_returns_matches_upstream_rloo_construction():
    token_rewards = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [3.0, 1.0, 1.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ]
    )

    returns = compute_prime_process_returns(
        token_rewards,
        response_mask,
        np.asarray(["prompt", "prompt"], dtype=object),
    )

    expected = torch.tensor(
        [
            [-1.0 / 3.0, 5.0 / 6.0, 0.0],
            [1.0 / 2.0, -7.0 / 3.0, -7.0 / 6.0],
        ]
    )
    torch.testing.assert_close(returns, expected)


def test_prime_process_returns_groups_by_uid_not_row_position():
    token_rewards = torch.tensor(
        [
            [1.0, 0.0],
            [10.0, 0.0],
            [3.0, 0.0],
            [14.0, 0.0],
        ]
    )
    response_mask = torch.ones_like(token_rewards, dtype=torch.long)

    returns = compute_prime_process_returns(
        token_rewards,
        response_mask,
        np.asarray(["a", "b", "a", "b"], dtype=object),
    )

    # Changing group b cannot affect either member of group a.
    changed = token_rewards.clone()
    changed[[1, 3]] += 1000
    changed_returns = compute_prime_process_returns(
        changed,
        response_mask,
        np.asarray(["a", "b", "a", "b"], dtype=object),
    )
    torch.testing.assert_close(returns[[0, 2]], changed_returns[[0, 2]])


def test_prime_process_returns_rejects_singleton_groups():
    with pytest.raises(ValueError, match="at least two responses"):
        compute_prime_process_returns(
            torch.ones(1, 2),
            torch.ones(1, 2),
            np.asarray(["only"], dtype=object),
        )
