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

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import compute_advantage


def test_compute_advantage_dispatches_normalized_rloo_from_string_name():
    rewards = torch.zeros(8, 3)
    rewards[:, -1] = torch.tensor([1.0] + [-1.0] * 7)
    batch = DataProto.from_dict(
        tensors={
            "token_level_rewards": rewards,
            "response_mask": torch.ones_like(rewards),
        },
        non_tensors={"uid": np.array(["prompt"] * 8, dtype=object)},
    )

    compute_advantage(
        batch,
        adv_estimator="normalized_rloo",
        config=OmegaConf.create({"norm_adv_by_std_in_grpo": True}),
    )

    assert batch.batch["advantages"][0, 0].item() == pytest.approx(2.82842, abs=1e-5)
    torch.testing.assert_close(batch.batch["returns"], batch.batch["advantages"])
    assert AdvantageEstimator.NORMALIZED_RLOO.value == "normalized_rloo"
