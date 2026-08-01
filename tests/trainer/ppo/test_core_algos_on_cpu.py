# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import random
import unittest

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    RatioValueCritic,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_length_adaptive_gae_advantage_return,
    compute_log_prob_values,
    compute_my_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    compute_vinfo_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def test_length_adaptive_gae_matches_per_response_lambda():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    values = torch.tensor([[0.2, 0.4, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

    actual_advantages, actual_returns = compute_length_adaptive_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=0.9,
        alpha=1.0,
    )
    expected_advantages, expected_returns = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=0.9,
        lam=torch.tensor([0.5, 0.75]),
    )

    torch.testing.assert_close(actual_advantages, expected_advantages)
    torch.testing.assert_close(actual_returns, expected_returns)


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_length_adaptive_gae_rejects_nonpositive_alpha(alpha):
    ones = torch.ones(1, 2)
    with pytest.raises(ValueError, match="alpha > 0"):
        compute_length_adaptive_gae_advantage_return(
            token_level_rewards=ones,
            values=ones,
            response_mask=ones,
            gamma=1.0,
            alpha=alpha,
        )


def test_length_adaptive_gae_clamps_scaled_length_to_one():
    ones = torch.ones(1, 2)
    actual = compute_length_adaptive_gae_advantage_return(
        token_level_rewards=ones,
        values=ones,
        response_mask=ones,
        gamma=1.0,
        alpha=0.4,
    )
    expected = compute_gae_advantage_return(
        token_level_rewards=ones,
        values=ones,
        response_mask=ones,
        gamma=1.0,
        lam=0.0,
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_length_adaptive_gae_rejects_empty_response():
    zeros = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="at least one valid token"):
        compute_length_adaptive_gae_advantage_return(
            token_level_rewards=zeros,
            values=zeros,
            response_mask=zeros,
            gamma=1.0,
            alpha=1.0,
        )


def test_compute_log_prob_values():
    ratio = torch.tensor(
        [
            [0.2, 0.3, 0.4, 99.0],
            [99.0, -0.1, 0.5, 0.2],
            [0.7, -0.2, 0.1, 99.0],
            [-0.4, 0.6, 0.3, 99.0],
        ]
    )
    old_log_prob = ratio.clone().requires_grad_(True)
    ref_log_prob = torch.zeros_like(ratio, requires_grad=True)
    token_level_rewards = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    group_ids = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"], dtype=object)

    values, actual_target = compute_log_prob_values(
        old_log_prob,
        ref_log_prob,
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        a=torch.tensor(1.0),
        b=torch.tensor(0.0),
        group_ids=group_ids,
    )

    expected_values = torch.tensor(
        [
            [-1.0, torch.tanh(torch.tensor(0.2)), torch.tanh(torch.tensor(0.5)), 0.0],
            [0.0, 1.0, torch.tanh(torch.tensor(-0.1)), torch.tanh(torch.tensor(0.4))],
            [-1.0, torch.tanh(torch.tensor(0.7)), torch.tanh(torch.tensor(0.5)), 0.0],
            [1.0, torch.tanh(torch.tensor(-0.4)), torch.tanh(torch.tensor(0.2)), 0.0],
        ]
    )
    expected_target = token_level_rewards.sum(dim=-1, keepdim=True).expand_as(ratio)

    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(actual_target, expected_target)


def test_compute_log_prob_values_singleton_group_keeps_tanh_b_at_v0():
    ratio = torch.tensor([[99.0, 0.4, 0.2]])
    response_mask = torch.tensor([[0.0, 1.0, 1.0]])

    values, _ = compute_log_prob_values(
        old_log_prob=ratio,
        ref_log_prob=torch.zeros_like(ratio),
        token_level_rewards=torch.tensor([[0.0, 0.0, 1.0]]),
        response_mask=response_mask,
        a=torch.tensor(2.0),
        b=torch.tensor(0.25),
        group_ids=np.array(["singleton"], dtype=object),
    )

    expected = torch.tensor(
        [[0.0, torch.tanh(torch.tensor(0.25)), torch.tanh(torch.tensor(1.05))]]
    )
    torch.testing.assert_close(values, expected)


def test_ratio_value_critic_initialization():
    critic = RatioValueCritic()

    torch.testing.assert_close(critic.a, torch.tensor(1.0))
    torch.testing.assert_close(critic.b, torch.tensor(0.0))


def test_ratio_value_critic_adamw_update_reduces_masked_mse():
    critic = RatioValueCritic()
    optimizer = torch.optim.AdamW(critic.parameters(), lr=0.1, weight_decay=0.01)
    ratio = torch.tensor([[-0.2, -0.1], [0.1, 0.2]])
    old_log_prob = ratio - 1.0
    ref_log_prob = torch.full_like(ratio, -1.0)
    token_level_rewards = torch.tensor([[0.0, -1.0], [0.0, 1.0]])
    response_mask = torch.ones_like(ratio)

    loss_before, _, _ = critic.loss(
        old_log_prob,
        ref_log_prob,
        token_level_rewards,
        response_mask,
        group_ids=np.array(["prompt", "prompt"], dtype=object),
    )
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()
    loss_after, _, _ = critic.loss(
        old_log_prob,
        ref_log_prob,
        token_level_rewards,
        response_mask,
        group_ids=np.array(["prompt", "prompt"], dtype=object),
    )

    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert loss_after < loss_before


def test_ratio_value_critic_loss_ignores_masked_tokens():
    critic = RatioValueCritic()
    old_log_prob = torch.tensor([[0.0, 0.0, 100.0], [0.0, 0.0, -100.0]])
    ref_log_prob = torch.zeros_like(old_log_prob)
    token_level_rewards = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    value_loss, values, target = critic.loss(
        old_log_prob,
        ref_log_prob,
        token_level_rewards,
        response_mask,
        group_ids=np.array(["prompt", "prompt"], dtype=object),
    )

    # LOO V0 values are deliberately very different from their own targets.
    # The loss is exactly 1 because only the second valid token participates.
    torch.testing.assert_close(values, torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    torch.testing.assert_close(target, torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]))
    torch.testing.assert_close(value_loss, torch.tensor(1.0))


def test_compute_log_prob_values_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must have the same shape"):
        compute_log_prob_values(
            torch.zeros(2, 3),
            torch.zeros(2, 4),
            token_level_rewards=torch.zeros(2, 3),
            response_mask=torch.ones(2, 3),
            a=torch.tensor(1.0),
            b=torch.tensor(0.0),
        )


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_discounted_future_mean(gamma: float):
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [-1.0, 0.5, 2.0, -3.0, 4.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    expected_sum = torch.zeros_like(values)
    expected_weight = torch.zeros_like(values)
    for batch_idx in range(values.shape[0]):
        for token_idx in range(values.shape[1]):
            for future_idx in range(token_idx, values.shape[1]):
                discount = gamma ** (future_idx - token_idx) * response_mask[batch_idx, future_idx]
                expected_sum[batch_idx, token_idx] += discount * values[batch_idx, future_idx]
                expected_weight[batch_idx, token_idx] += discount
    expected = torch.where(expected_weight > 0, expected_sum / expected_weight, 0.0)
    expected *= response_mask

    actual = verl.trainer.ppo.core_algos._discounted_future_mean(
        values,
        response_mask,
        gamma=gamma,
        chunk_size=2,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("gamma", [-0.1, 1.1])
def test_discounted_future_mean_rejects_invalid_gamma(gamma: float):
    values = torch.ones(1, 2)
    with pytest.raises(ValueError, match=r"gamma must be in \[0, 1\]"):
        verl.trainer.ppo.core_algos._discounted_future_mean(values, values, gamma=gamma)


def test_compute_policy_loss_my_rejects_nonpositive_tau():
    config = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss": {"gamma": 0.99, "tau": 0.0, "chunk_size": 2},
        }
    )
    zeros = torch.zeros(1, 2)
    ones = torch.ones(1, 2)

    with pytest.raises(ValueError, match="tau must be positive"):
        verl.trainer.ppo.core_algos.compute_policy_loss_my(
            old_log_prob=zeros,
            log_prob=zeros,
            ref_log_prob=zeros,
            entropy=ones,
            advantages=ones,
            response_mask=ones,
            config=config,
        )


def test_compute_my_outcome_advantage_returns_token_weights():
    response_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    token_level_rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    old_log_prob = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ref_log_prob = torch.zeros_like(old_log_prob)
    entropy = torch.tensor([[1.0, 2.0], [2.0, 1.0]])

    advantages, returns, w = compute_my_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=np.array(["group", "group"]),
        old_log_prob=old_log_prob,
        ref_log_prob=ref_log_prob,
        entropy=entropy,
    )

    assert advantages.shape == returns.shape == w.shape == response_mask.shape
    assert torch.equal(advantages, returns)
    assert torch.isfinite(w[response_mask.bool()]).all()
    assert w[response_mask.bool()].max() > w[response_mask.bool()].min()


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def test_vinfo_returns_unmodified_grpo_advantage_and_weights():
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    token_level_rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    index = np.array([0, 0])
    answer_log_prob = torch.tensor([[0.0, 0.2, 0.4, 0.6], [0.0, -0.2, -0.4, -0.6]])
    answer_step_end_mask = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.bool)

    expected_advantages, expected_returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    advantages, returns, w = compute_vinfo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        answer_log_prob=answer_log_prob,
        answer_step_end_mask=answer_step_end_mask,
    )

    assert torch.equal(advantages, expected_advantages)
    assert torch.equal(returns, expected_returns)
    assert w.shape == advantages.shape


if __name__ == "__main__":
    unittest.main()
