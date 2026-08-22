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

from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager, RewardComputeWorker  # noqa: F401


@register("remote")
class RemoteRewardManager(DAPORewardManager):
    """Compatibility alias; DAPO now uses Ray scorer processes directly."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)
        assert not self.is_async_reward_score, "Async reward score is not supported in remote reward manager. "
