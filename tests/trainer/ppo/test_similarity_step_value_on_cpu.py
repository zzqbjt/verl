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
import torch

from verl.trainer.ppo.step_value_similarity import compute_similarity_step_values
from verl.utils.step_split import build_step_start_mask


def _two_step_masks(batch_size: int):
    response_mask = torch.ones(batch_size, 2, dtype=torch.bool)
    step_end_mask = torch.ones_like(response_mask)
    step_start_mask = build_step_start_mask(step_end_mask, response_mask)
    return response_mask, step_start_mask, step_end_mask


def test_similarity_provider_uses_raw_outcomes_and_exact_terminal_values():
    response_mask, starts, ends = _two_step_masks(3)
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, 1.0]],
        ]
    )
    outcomes = torch.tensor([-2.0, 3.0, 5.0])

    values, metrics = compute_similarity_step_values(
        packed_step_embeddings=embeddings,
        step_start_mask=starts,
        step_end_mask=ends,
        response_mask=response_mask,
        outcomes=outcomes,
        prompt_ids=np.array(["p", "p", "p"], dtype=object),
        top_k=1,
        tau=0.002,
        position_window=1.0,
        iterations=1,
    )

    assert values[0, 0] == 3.0
    assert values[1, 0] == -2.0
    torch.testing.assert_close(values[:, 1], outcomes)
    assert metrics["similarity_step_value/retrieval_coverage"] == 1.0


def test_similarity_iterations_propagate_previous_step_values_synchronously():
    response_mask, starts, ends = _two_step_masks(2)
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    outcomes = torch.tensor([-2.0, 3.0])

    values, _ = compute_similarity_step_values(
        packed_step_embeddings=embeddings,
        step_start_mask=starts,
        step_end_mask=ends,
        response_mask=response_mask,
        outcomes=outcomes,
        prompt_ids=np.array(["p", "p"], dtype=object),
        top_k=1,
        tau=0.002,
        position_window=1.0,
        iterations=2,
    )

    torch.testing.assert_close(values[:, 0], outcomes)
    torch.testing.assert_close(values[:, 1], outcomes)


def test_similarity_position_fallback_excludes_the_target_response():
    response_mask = torch.ones(2, 6, dtype=torch.bool)
    ends = torch.tensor(
        [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1]],
        dtype=torch.bool,
    )
    starts = build_step_start_mask(ends, response_mask)
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    outcomes = torch.tensor([-2.0, 3.0])

    values, metrics = compute_similarity_step_values(
        packed_step_embeddings=embeddings,
        step_start_mask=starts,
        step_end_mask=ends,
        response_mask=response_mask,
        outcomes=outcomes,
        prompt_ids=np.array(["p", "p"], dtype=object),
        top_k=1,
        tau=0.002,
        position_window=0.01,
        iterations=1,
    )

    assert values[0, 0] == 3.0
    assert values[1, 4] == -2.0
    assert metrics["similarity_step_value/retrieval_coverage"] == 0.0
