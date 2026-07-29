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

import pytest
import torch

from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import ActorConfig


def _make_actor(alpha: float, initial_value: float = 0.0) -> DataParallelPPOActor:
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = torch.nn.Linear(2, 1, bias=False)
    actor.ema_reference_module = torch.nn.Linear(2, 1, bias=False)
    actor.actor_module.weight.data.fill_(initial_value)
    actor.ref_ema_alpha = alpha
    actor._copy_actor_to_ema_reference()
    return actor


def test_reference_uses_requested_ema_update():
    actor = _make_actor(alpha=0.25)
    actor.actor_module.weight.data.fill_(2.0)

    actor.update_ema_reference()

    torch.testing.assert_close(actor.ema_reference_module.weight, torch.full((1, 2), 0.5))
    torch.testing.assert_close(actor.actor_module.weight, torch.full((1, 2), 2.0))


def test_reference_context_restores_online_actor_after_error():
    actor = _make_actor(alpha=0.25)
    actor.actor_module.weight.data.fill_(2.0)
    actor.update_ema_reference()

    with pytest.raises(RuntimeError, match="forward failed"):
        with actor._use_ema_reference():
            torch.testing.assert_close(actor.actor_module.weight, torch.full((1, 2), 0.5))
            raise RuntimeError("forward failed")

    torch.testing.assert_close(actor.actor_module.weight, torch.full((1, 2), 2.0))


def test_reference_state_dict_round_trip():
    actor = _make_actor(alpha=0.25)
    actor.actor_module.weight.data.fill_(2.0)
    actor.update_ema_reference()
    state_dict = actor.ema_reference_state_dict()

    restored_actor = _make_actor(alpha=0.25, initial_value=9.0)
    restored_actor.load_ema_reference_state_dict(state_dict)

    torch.testing.assert_close(restored_actor.ema_reference_module.weight, torch.full((1, 2), 0.5))


def test_missing_reference_checkpoint_resets_reference_from_loaded_actor():
    actor = _make_actor(alpha=0.25)
    actor.actor_module.weight.data.fill_(3.0)

    actor.load_ema_reference_state_dict(None)

    torch.testing.assert_close(actor.ema_reference_module.weight, torch.full((1, 2), 3.0))


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_reference_alpha_must_be_a_convex_weight(alpha: float):
    with pytest.raises(ValueError, match="ema_alpha must be in"):
        ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            use_dynamic_bsz=True,
            ema_alpha=alpha,
        )
