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

from omegaconf import OmegaConf

from verl.trainer.ppo.utils import need_critic, need_reference_policy


def test_dgpo_needs_reference_policy():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {"loss_mode": "dgpo"},
                    "use_kl_loss": False,
                    "kl_loss_coef": 0.0,
                }
            },
            "algorithm": {"use_kl_in_reward": False},
        }
    )

    assert need_reference_policy(config)


def _make_gae_config(critic_enable, adv_estimator="gae"):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {"loss_mode": "vanilla"},
                    "use_kl_loss": False,
                    "kl_loss_coef": 0.0,
                }
            },
            "algorithm": {
                "adv_estimator": adv_estimator,
                "use_kl_in_reward": False,
            },
            "critic": {"enable": critic_enable},
        }
    )


def test_critic_free_gae_needs_reference_policy():
    assert need_reference_policy(_make_gae_config(critic_enable=False))


def test_critic_free_length_adaptive_gae_needs_reference_policy():
    config = _make_gae_config(critic_enable=False, adv_estimator="length_adaptive_gae")
    assert need_reference_policy(config)


def test_standard_gae_does_not_need_reference_policy():
    assert not need_reference_policy(_make_gae_config(critic_enable=None))


def test_length_adaptive_gae_uses_standard_critic_by_default():
    config = _make_gae_config(critic_enable=None, adv_estimator="length_adaptive_gae")
    assert need_critic(config)
