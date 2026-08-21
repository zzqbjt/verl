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

import torch

from verl.workers.actor.dp_actor import DataParallelPPOActor


def test_answer_log_prob_maps_shared_endpoints_to_prefix_lengths():
    response_mask = torch.tensor([0, 1, 1, 0, 1, 1, 0], dtype=torch.bool)
    step_end_mask = torch.tensor([0, 0, 1, 0, 0, 1, 0], dtype=torch.bool)

    positions, prefix_lengths = DataParallelPPOActor._shared_step_prefixes(response_mask, step_end_mask)

    assert positions == [2, 5]
    assert prefix_lengths == [2, 4]
