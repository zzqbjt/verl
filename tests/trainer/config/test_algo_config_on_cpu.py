# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.config import AlgoConfig, KLControlConfig, StepSplitConfig, StepValueAdvantageConfig
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    get_adv_estimator_fn,
)
from verl.utils.config import omega_conf_to_dataclass


class TestAlgoConfig(unittest.TestCase):
    """Test the AlgoConfig dataclass and its integration with core algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample algorithm config as DictConfig (similar to what comes from YAML)
        self.config_dict = {
            "_target_": "verl.trainer.config.AlgoConfig",
            "gamma": 0.99,
            "lam": 0.95,
            "adv_estimator": "gae",
            "norm_adv_by_std_in_grpo": True,
            "step_value": {
                "_target_": "verl.trainer.config.StepValueAdvantageConfig",
                "provider": "similarity",
                "lam": 0.8,
                "norm_by_group_std": False,
                "target_key": "acc",
                "task_reward_key": "score",
            },
            "use_kl_in_reward": True,
            "kl_penalty": "kl",
            "kl_ctrl": {
                "_target_": "verl.trainer.config.KLControlConfig",
                "type": "adaptive",
                "kl_coef": 0.002,
                "horizon": 5000,
                "target_kl": 0.05,
            },
            "use_pf_ppo": True,
            "pf_ppo": {"reweight_method": "max_min", "weight_pow": 3.0},
        }
        self.omega_config = OmegaConf.create(self.config_dict)

    def test_dataclass_creation_from_dict(self):
        """Test creating AlgoConfig from dictionary."""
        config = omega_conf_to_dataclass(self.config_dict)

        self.assertIsInstance(config, AlgoConfig)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.lam, 0.95)
        self.assertEqual(config.adv_estimator, "gae")
        self.assertTrue(config.norm_adv_by_std_in_grpo)
        self.assertTrue(config.use_kl_in_reward)
        self.assertEqual(config.kl_penalty, "kl")
        self.assertTrue(config.use_pf_ppo)

    def test_dataclass_creation_from_omega_config(self):
        """Test creating AlgoConfig from OmegaConf DictConfig."""
        config = omega_conf_to_dataclass(self.omega_config)

        self.assertIsInstance(config, AlgoConfig)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.lam, 0.95)

    def test_nested_configs(self):
        """Test that nested configurations are properly converted."""
        config = omega_conf_to_dataclass(self.omega_config)

        # Test KL control config
        self.assertIsInstance(config.kl_ctrl, KLControlConfig)
        self.assertEqual(config.kl_ctrl.type, "adaptive")
        self.assertEqual(config.kl_ctrl.kl_coef, 0.002)
        self.assertEqual(config.kl_ctrl.horizon, 5000)
        self.assertEqual(config.kl_ctrl.target_kl, 0.05)

        # Test step-value advantage config
        self.assertIsInstance(config.step_value, StepValueAdvantageConfig)
        self.assertEqual(config.step_value.provider, "similarity")
        self.assertEqual(config.step_value.lam, 0.8)
        self.assertFalse(config.step_value.norm_by_group_std)
        self.assertEqual(config.step_value.target_key, "acc")
        self.assertEqual(config.step_value.task_reward_key, "score")

        # Test PF PPO config
        self.assertEqual(config.pf_ppo.get("reweight_method"), "max_min")
        self.assertEqual(config.pf_ppo.get("weight_pow"), 3.0)

    def test_default_values(self):
        """Test that default values are properly set."""
        minimal_config = {"gamma": 0.8}
        config = omega_conf_to_dataclass(minimal_config, AlgoConfig)

        self.assertEqual(config.gamma, 0.8)
        self.assertEqual(config.lam, 1.0)  # default value
        self.assertEqual(config.length_adaptive_gae_alpha, 1.0)
        self.assertEqual(config.adv_estimator, "gae")  # default value
        self.assertTrue(config.norm_adv_by_std_in_grpo)  # default value
        self.assertIsInstance(config.step_split, StepSplitConfig)
        self.assertFalse(config.step_split.enabled)
        self.assertEqual(config.step_split.lookahead_tokens, 10)
        self.assertFalse(config.step_split.separate_preamble)
        self.assertIsInstance(config.step_value, StepValueAdvantageConfig)
        self.assertEqual(config.step_value.provider, "probe")
        self.assertEqual(config.step_value.lam, 0.9)
        self.assertTrue(config.step_value.norm_by_group_std)
        self.assertTrue(config.step_value.zero_when_group_uniform)
        self.assertEqual(config.step_value.target_key, "acc")
        self.assertEqual(config.step_value.task_reward_key, "score")
        self.assertFalse(config.step_value.prompt_center_calibration_enabled)
        self.assertEqual(config.step_value.prompt_center_calibration_slope, 1.0)
        self.assertEqual(config.step_value.prompt_center_calibration_intercept, 0.0)
        self.assertFalse(config.step_value.prompt_center_audit_enabled)
        self.assertEqual(config.step_value.prompt_center_audit_groups, 16)
        self.assertEqual(config.step_value.prompt_center_audit_window, 2)
        self.assertEqual(config.step_value.prompt_center_audit_seed, 0)
        self.assertEqual(config.step_value.similarity_top_k, 3)
        self.assertEqual(config.step_value.similarity_tau, 0.002)
        self.assertEqual(config.step_value.similarity_position_window, 0.2)
        self.assertEqual(config.step_value.similarity_iterations, 1)
        self.assertIsNone(config.step_value.lookahead_tokens)
        self.assertIsNone(config.step_value.separate_preamble)
        self.assertEqual(config.ratio_value_critic["a_init"], 1.0)
        self.assertEqual(config.ratio_value_critic["weight_decay"], 1e-2)
        self.assertFalse(config.use_kl_in_reward)  # default value
        self.assertEqual(config.kl_penalty, "kl")  # default value
        self.assertFalse(config.use_pf_ppo)  # default value

    def test_get_method_backward_compatibility(self):
        """Test the get method for backward compatibility."""
        config = omega_conf_to_dataclass(self.omega_config)

        # Test existing attribute
        self.assertEqual(config.get("gamma"), 0.99)
        self.assertEqual(config.get("gamma", 1.0), 0.99)

        # Test non-existing attribute
        self.assertIsNone(config.get("non_existing"))
        self.assertEqual(config.get("non_existing", "default"), "default")

    def test_post_init_nested_configs(self):
        """Test that __post_init__ properly initializes nested configs when None."""
        # Create config without nested configs
        minimal_config = AlgoConfig(gamma=0.9)

        # Check that nested configs are initialized
        self.assertIsNotNone(minimal_config.kl_ctrl)
        self.assertIsInstance(minimal_config.kl_ctrl, KLControlConfig)
        assert not minimal_config.pf_ppo

    def test_config_init_from_yaml(self):
        import os

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
            cfg = compose(config_name="ppo_trainer")
        algo_config = omega_conf_to_dataclass(cfg.algorithm)
        from verl.trainer.config import AlgoConfig

        assert isinstance(algo_config, AlgoConfig)
        assert isinstance(algo_config.step_split, StepSplitConfig)
        assert isinstance(algo_config.step_value, StepValueAdvantageConfig)

    def test_step_split_config_validation(self):
        """Step splitting is independently configurable without a probe."""
        config = StepSplitConfig(enabled=True, lookahead_tokens=4, separate_preamble=True)
        self.assertTrue(config.enabled)
        self.assertEqual(config.lookahead_tokens, 4)
        self.assertTrue(config.separate_preamble)

        invalid_cases = [
            ({"enabled": 1}, "step_split.enabled"),
            ({"lookahead_tokens": 0}, "step_split.lookahead_tokens"),
            ({"separate_preamble": 1}, "step_split.separate_preamble"),
        ]
        for kwargs, expected_message in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, expected_message):
                StepSplitConfig(**kwargs)

    def test_step_value_advantage_config_validation(self):
        """Reject invalid step-value advantage settings."""
        invalid_cases = [
            ({"provider": "unknown"}, "step_value.provider"),
            ({"lam": -0.1}, "step_value.lam"),
            ({"lam": 1.1}, "step_value.lam"),
            ({"norm_by_group_std": 1}, "step_value.norm_by_group_std"),
            ({"zero_when_group_uniform": 1}, "step_value.zero_when_group_uniform"),
            ({"target_key": "  "}, "step_value.target_key"),
            ({"task_reward_key": "  "}, "step_value.task_reward_key"),
            ({"prompt_center_calibration_enabled": 1}, "step_value.prompt_center_calibration_enabled"),
            ({"prompt_center_calibration_slope": 0.0}, "step_value.prompt_center_calibration_slope"),
            ({"prompt_center_calibration_slope": float("inf")}, "step_value.prompt_center_calibration_slope"),
            (
                {"prompt_center_calibration_intercept": float("nan")},
                "step_value.prompt_center_calibration_intercept",
            ),
            ({"prompt_center_audit_enabled": 1}, "step_value.prompt_center_audit_enabled"),
            ({"prompt_center_audit_groups": 0}, "step_value.prompt_center_audit_groups"),
            ({"prompt_center_audit_window": 0}, "step_value.prompt_center_audit_window"),
            ({"prompt_center_audit_seed": -1}, "step_value.prompt_center_audit_seed"),
            (
                {"prompt_center_audit_enabled": True, "prompt_center_calibration_enabled": True},
                "mutually exclusive",
            ),
            ({"similarity_top_k": 0}, "step_value.similarity_top_k"),
            ({"similarity_tau": 0.0}, "step_value.similarity_tau"),
            ({"similarity_position_window": 1.1}, "step_value.similarity_position_window"),
            ({"similarity_iterations": 0}, "step_value.similarity_iterations"),
            ({"lookahead_tokens": 0}, "step_value.lookahead_tokens"),
            ({"separate_preamble": 1}, "step_value.separate_preamble"),
        ]
        for kwargs, expected_message in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, expected_message):
                StepValueAdvantageConfig(**kwargs)


class TestAlgoCompute(unittest.TestCase):
    """Test the AlgoConfig dataclass and its integration with core algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        self.algo_config = AlgoConfig(
            gamma=0.99,
            lam=0.95,
            adv_estimator="gae",
            norm_adv_by_std_in_grpo=True,
            use_kl_in_reward=True,
            kl_penalty="kl",
            kl_ctrl=KLControlConfig(type="adaptive", kl_coef=0.002, horizon=5000, target_kl=0.05),
            use_pf_ppo=True,
            pf_ppo={"reweight_method": "max_min", "weight_pow": 3.0},
        )

    def test_advantage_estimator_with_cfg(self):
        """Test integration with advantage estimators from core_algos."""
        config = self.algo_config

        # Test GAE advantage estimator
        adv_fn = get_adv_estimator_fn(config.adv_estimator)
        self.assertIsNotNone(adv_fn)

        # Test with actual GAE computation
        batch_size, seq_len = 2, 5
        token_level_rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)

        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=config.gamma,
            lam=config.lam,
        )

        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))

    def test_grpo_advantage_estimator_with_cfg(self):
        """Test integration with GRPO advantage estimator."""
        grpo_config = AlgoConfig(adv_estimator="grpo", norm_adv_by_std_in_grpo=True)

        # Test GRPO advantage computation
        batch_size, seq_len = 4, 3
        token_level_rewards = torch.tensor([[1.0, 0.5, 0.0], [2.0, 1.0, 0.0], [0.5, 0.2, 0.0], [1.5, 0.8, 0.0]])
        response_mask = torch.ones(batch_size, seq_len)
        index = np.array([0, 0, 1, 1])  # Two groups

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=grpo_config.norm_adv_by_std_in_grpo,
        )

        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))


if __name__ == "__main__":
    unittest.main()
