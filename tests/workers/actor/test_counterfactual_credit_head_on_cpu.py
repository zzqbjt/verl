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
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.workers.actor.dp_actor import CounterfactualCreditHead, DataParallelPPOActor
from verl.workers.config import CounterfactualCreditHeadConfig, FSDPActorConfig


class _TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, vision_config=None)
        self.embed_tokens = torch.nn.Embedding(32, 4)
        self.lm_head = torch.nn.Linear(4, 32, bias=False)
        with torch.no_grad():
            values = torch.arange(32, dtype=torch.float32).unsqueeze(-1) / 10.0
            self.embed_tokens.weight.copy_(values.repeat(1, 4))

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, **kwargs):
        assert "output_hidden_states" not in kwargs
        hidden = self.embed_tokens(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _make_actor(use_remove_padding: bool, *, use_dynamic_bsz: bool = False) -> DataParallelPPOActor:
    config = FSDPActorConfig(
        strategy="fsdp",
        rollout_n=1,
        use_dynamic_bsz=use_dynamic_bsz,
        ppo_micro_batch_size_per_gpu=2,
        use_remove_padding=use_remove_padding,
        use_torch_compile=False,
        ulysses_sequence_parallel_size=1,
        counterfactual_credit_head=CounterfactualCreditHeadConfig(
            enabled=True,
            hidden_dim=5,
            lr=1e-2,
            weight_decay=0.0,
        ),
    )
    model = _TinyCausalLM()
    with patch("torch.distributed.get_rank", return_value=0):
        actor = DataParallelPPOActor(config, model, torch.optim.SGD(model.parameters(), lr=1e-3))
    actor.device_name = "cpu"
    actor._counterfactual_credit_needs_broadcast = False
    return actor


def _make_data(global_step: int = 1) -> DataProto:
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [0, 6, 7, 8, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], dtype=torch.long)
    step_end_mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    anchor_mask = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool)
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": attention_mask.cumsum(dim=-1) - 1,
            "responses": input_ids[:, -3:],
            "step_end_mask": step_end_mask,
            "credit_anchor_mask": anchor_mask,
            "credit_anchor_targets": torch.tensor([[0.5, 0.0, 0.0], [0.0, -0.5, 0.0]]),
            "credit_anchor_weights": anchor_mask.float(),
        },
        batch_size=[2],
    )
    return DataProto(
        batch=batch,
        meta_info={
            "micro_batch_size": 2,
            "temperature": 1.0,
            "use_dynamic_bsz": False,
            "global_steps": global_step,
            "huber_delta": 1.0,
        },
    )


def _cpu_logprobs(logits, labels, **_):
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("use_remove_padding", [False, True])
def test_credit_head_uses_prompt_and_previous_step_boundaries_and_updates(use_remove_padding: bool):
    actor = _make_actor(use_remove_padding)
    assert isinstance(actor.counterfactual_credit_head, CounterfactualCreditHead)
    assert actor.counterfactual_credit_head.input_layer.in_features == 8
    assert actor.counterfactual_credit_head.input_layer.out_features == 5
    initial = {
        name: parameter.detach().clone() for name, parameter in actor.counterfactual_credit_head.named_parameters()
    }

    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
    ):
        outputs = actor.compute_counterfactual_credit(_make_data())

    assert outputs["credit_predictions"].shape == (2, 3)
    assert torch.equal(outputs["credit_predictions"] != 0, _make_data().batch["step_end_mask"])
    assert torch.isfinite(outputs["credit_head_loss"]).all()
    assert torch.all(
        (outputs["credit_head_direction_agreement"] >= 0) & (outputs["credit_head_direction_agreement"] <= 1)
    )
    assert "credit_head_updates" not in outputs
    assert any(
        not torch.equal(initial[name], parameter)
        for name, parameter in actor.counterfactual_credit_head.named_parameters()
    )


def test_credit_representation_is_detached_h_pre_and_post_delta():
    boundary_hidden = torch.tensor([[1.0, 2.0], [4.0, 8.0], [5.0, 10.0], [2.0, 3.0], [7.0, 9.0]])
    representation = DataParallelPPOActor.build_counterfactual_credit_representations(
        boundary_hidden,
        torch.tensor([2, 1]),
    )
    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0, 6.0],
            [4.0, 8.0, 1.0, 2.0],
            [2.0, 3.0, 5.0, 6.0],
        ]
    )
    torch.testing.assert_close(representation, expected)
    assert not representation.requires_grad


def test_dynamic_credit_micro_batches_restore_original_response_order():
    actor = _make_actor(True, use_dynamic_bsz=True)
    data = _make_data()
    data.meta_info["use_dynamic_bsz"] = True
    data.meta_info["max_token_len"] = 5
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
    ):
        outputs = actor.compute_counterfactual_credit(data)

    assert torch.equal(outputs["credit_predictions"] != 0, data.batch["step_end_mask"])
    assert "credit_head_direction_agreement" in outputs


def test_credit_head_direction_agreement_ignores_zero_targets():
    actor = _make_actor(False)
    with torch.no_grad():
        for parameter in actor.counterfactual_credit_head.parameters():
            parameter.zero_()
        actor.counterfactual_credit_head.output_layer.bias.fill_(0.25)
    data = _make_data()
    data.batch["credit_anchor_targets"][1, 1] = 0.0
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
    ):
        outputs = actor.compute_counterfactual_credit(data)

    torch.testing.assert_close(outputs["credit_head_direction_agreement"], torch.ones(2))


def test_credit_head_checkpoint_round_trip():
    actor = _make_actor(False)
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
    ):
        actor.compute_counterfactual_credit(_make_data())
    state = actor.counterfactual_credit_state_dict()

    restored = _make_actor(False)
    restored.load_counterfactual_credit_state_dict(state)

    assert restored.counterfactual_credit_updates == 1
    assert restored.counterfactual_credit_last_global_step == 1
    for expected, actual in zip(
        actor.counterfactual_credit_head.parameters(),
        restored.counterfactual_credit_head.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
