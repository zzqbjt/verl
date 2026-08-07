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

from types import SimpleNamespace

import pytest
import torch

from verl.utils.fsdp_utils import get_fsdp_checkpoint_shard_info


def _mesh(values, dim_names):
    return SimpleNamespace(
        mesh=torch.tensor(values),
        mesh_dim_names=dim_names,
    )


@pytest.mark.parametrize("rank", range(4))
def test_full_fsdp_mesh_keeps_every_shard(rank):
    mesh = _mesh([0, 1, 2, 3], ("fsdp",))

    assert get_fsdp_checkpoint_shard_info(mesh, rank) == (4, rank, rank)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, (2, 0, 0)),
        (1, (2, 1, 1)),
        (2, (2, 0, 0)),
        (3, (2, 1, 1)),
    ],
)
def test_hsdp_mesh_maps_replicas_to_first_fsdp_group(rank, expected):
    mesh = _mesh([[0, 1], [2, 3]], ("ddp", "fsdp"))

    assert get_fsdp_checkpoint_shard_info(mesh, rank) == expected


@pytest.mark.parametrize("rank", range(4))
def test_fsdp_size_one_writes_one_full_checkpoint(rank):
    mesh = _mesh([[0], [1], [2], [3]], ("ddp", "fsdp"))

    assert get_fsdp_checkpoint_shard_info(mesh, rank) == (1, 0, 0)


def test_unnamed_multidimensional_mesh_is_rejected():
    mesh = _mesh([[0, 1], [2, 3]], None)

    with pytest.raises(ValueError, match="Cannot identify the FSDP dimension"):
        get_fsdp_checkpoint_shard_info(mesh, 0)


def test_rank_must_exist_exactly_once():
    mesh = _mesh([0, 1], ("fsdp",))

    with pytest.raises(ValueError, match="must occur exactly once"):
        get_fsdp_checkpoint_shard_info(mesh, 2)
