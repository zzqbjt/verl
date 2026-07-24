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

from verl.trainer.ppo.utils import need_reference_policy


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
