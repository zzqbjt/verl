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
            "credit_anchor_q_targets": torch.tensor([[0.75, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "credit_anchor_v_targets": torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.5, 0.0]]),
            "credit_start_value_mask": torch.tensor([True, True]),
            "credit_start_value_train_mask": torch.tensor([True, False]),
            "credit_start_value_targets": torch.tensor([0.25, 0.5]),
            "credit_terminal_value_mask": torch.tensor([True, True]),
            "credit_terminal_value_targets": torch.tensor([1.0, 0.0]),
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
            "credit_value_loss_weights": (0.6, 0.2, 0.2),
        },
    )


def _cpu_logprobs(logits, labels, **_):
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("use_remove_padding", [False, True])
def test_credit_head_uses_prompt_and_previous_step_boundaries_and_updates(use_remove_padding: bool):
    actor = _make_actor(use_remove_padding)
    assert isinstance(actor.counterfactual_credit_head, CounterfactualCreditHead)
    assert actor.counterfactual_credit_head.input_layer.in_features == 4
    assert actor.counterfactual_credit_head.input_layer.out_features == 5
    assert actor.counterfactual_credit_head.output_layer.weight.any()
    assert actor.counterfactual_credit_head.output_layer.weight.std() < 0.01
    assert not actor.counterfactual_credit_head.output_layer.bias.any()
    initial = {
        name: parameter.detach().clone() for name, parameter in actor.counterfactual_credit_head.named_parameters()
    }

    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
    ):
        outputs = actor.compute_counterfactual_credit(_make_data())

    assert outputs["credit_predictions"].shape == (2, 3)
    expected_credit = torch.tensor([[0.5, 0.0, 0.25], [0.0, -0.5, 0.0]])
    torch.testing.assert_close(outputs["credit_predictions"], expected_credit)
    assert torch.isfinite(outputs["credit_head_value_bce"]).all()
    assert torch.isfinite(outputs["credit_head_difference_smooth_l1"]).all()
    torch.testing.assert_close(
        outputs["credit_head_total_loss"],
        outputs["credit_head_value_bce"] + 0.25 * outputs["credit_head_difference_smooth_l1"],
    )
    assert torch.all((outputs["credit_head_value_mae"] >= 0) & (outputs["credit_head_value_mae"] <= 1))
    assert torch.all(
        (outputs["credit_head_direction_agreement"] >= 0) & (outputs["credit_head_direction_agreement"] <= 1)
    )
    assert "credit_head_confident_direction_agreement" not in outputs
    assert "credit_head_confident_direction_coverage" not in outputs
    assert "credit_head_updates" not in outputs
    assert not torch.equal(
        initial["input_layer.weight"],
        actor.counterfactual_credit_head.input_layer.weight,
    )
    assert any(
        not torch.equal(initial[name], parameter)
        for name, parameter in actor.counterfactual_credit_head.named_parameters()
    )
    assert actor.counterfactual_credit_updates == 2


def test_value_boundaries_are_aligned_with_each_transition_and_detached():
    boundary_hidden = torch.tensor([[1.0, 2.0], [4.0, 8.0], [5.0, 10.0], [2.0, 3.0], [7.0, 9.0]])
    before, after, first, last, before_indices, after_indices = (
        DataParallelPPOActor.build_counterfactual_value_transitions(
            boundary_hidden,
            torch.tensor([2, 1]),
        )
    )
    torch.testing.assert_close(before, torch.tensor([[1.0, 2.0], [4.0, 8.0], [2.0, 3.0]]))
    torch.testing.assert_close(after, torch.tensor([[4.0, 8.0], [5.0, 10.0], [7.0, 9.0]]))
    assert torch.equal(first, torch.tensor([True, False, True]))
    assert torch.equal(last, torch.tensor([False, True, True]))
    assert torch.equal(before_indices, torch.tensor([0, 1, 3]))
    assert torch.equal(after_indices, torch.tensor([1, 2, 4]))
    assert not before.requires_grad
    assert not after.requires_grad


def test_credit_capture_uses_fixed_response_prefix_as_branch_start_boundary():
    actor = _make_actor(False)
    data = _make_data()
    model_inputs = {
        **data.batch,
        "credit_prefix_lengths": torch.tensor([0, 1], dtype=torch.long),
        "pad_token_id": 0,
    }
    with patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs):
        outputs = actor._forward_micro_batch(
            model_inputs,
            temperature=1.0,
            return_credit_boundary_hidden=True,
        )

    boundary_hidden = outputs["credit_boundary_hidden"]
    expected_token_values = torch.tensor([0.2, 0.3, 0.5, 0.7, 0.8]).unsqueeze(-1).repeat(1, 4)
    torch.testing.assert_close(boundary_hidden, expected_token_values)


def test_dynamic_credit_micro_batches_restore_original_response_order():
    actor = _make_actor(True, use_dynamic_bsz=True)
    with torch.no_grad():
        actor.counterfactual_credit_head.output_layer.bias.fill_(0.25)
    data = _make_data()
    data.meta_info["use_dynamic_bsz"] = True
    data.meta_info["max_token_len"] = 5
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
    ):
        outputs = actor.compute_counterfactual_credit(data)

    expected_credit = torch.tensor([[0.5, 0.0, 0.25], [0.0, -0.5, 0.0]])
    torch.testing.assert_close(outputs["credit_predictions"], expected_credit)
    assert "credit_head_direction_agreement" in outputs


def test_credit_head_direction_agreement_is_weighted_by_target_magnitude():
    actor = _make_actor(False)
    with torch.no_grad():
        for parameter in actor.counterfactual_credit_head.parameters():
            parameter.zero_()
        actor.counterfactual_credit_head.input_layer.weight[0, 0] = 1.0
        actor.counterfactual_credit_head.output_layer.weight[0, 0] = 1.0
    data = _make_data()
    data.batch["credit_anchor_targets"][1, 1] = -0.125
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
    ):
        outputs = actor.compute_counterfactual_credit(data)

    torch.testing.assert_close(outputs["credit_head_direction_agreement"], torch.full((2,), 0.8))


def test_credit_predictions_are_computed_before_current_batch_head_update():
    actor = _make_actor(False)
    with torch.no_grad():
        for parameter in actor.counterfactual_credit_head.parameters():
            parameter.zero_()
        actor.counterfactual_credit_head.output_layer.bias.fill_(0.25)

    def replace_head_during_optimizer_step():
        with torch.no_grad():
            actor.counterfactual_credit_head.output_layer.bias.fill_(0.75)

    data = _make_data()
    data.batch["credit_terminal_value_mask"][0] = False
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
        patch.object(
            actor.counterfactual_credit_optimizer,
            "step",
            side_effect=replace_head_during_optimizer_step,
        ),
    ):
        outputs = actor.compute_counterfactual_credit(data)

    old_value = torch.sigmoid(torch.tensor(0.25))
    expected = torch.tensor([[0.5, 0.0, old_value - 0.75], [0.0, -0.5, 0.0]])
    torch.testing.assert_close(outputs["credit_predictions"], expected)
    torch.testing.assert_close(actor.counterfactual_credit_head.output_layer.bias, torch.tensor([0.75]))


def test_zero_lambda_step_predicts_only_mc_anchors_for_probe_update():
    actor = _make_actor(False)
    data = _make_data()
    data.meta_info["predict_all_credit_steps"] = False
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
    ):
        outputs = actor.compute_counterfactual_credit(data)

    torch.testing.assert_close(outputs["credit_predictions"], torch.zeros_like(data.batch["step_end_mask"].float()))
    assert actor.counterfactual_credit_updates == 2
    assert torch.isfinite(outputs["credit_head_value_bce"]).all()
    assert torch.isfinite(outputs["credit_head_difference_smooth_l1"]).all()
    assert torch.isfinite(outputs["credit_head_total_loss"]).all()
    assert torch.isfinite(outputs["credit_head_value_mae"]).all()


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

    assert restored.counterfactual_credit_updates == 2
    assert restored.counterfactual_credit_last_global_step == 1
    for expected, actual in zip(
        actor.counterfactual_credit_head.parameters(),
        restored.counterfactual_credit_head.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_legacy_direct_credit_checkpoint_resets_only_the_probe():
    actor = _make_actor(False)
    actor.counterfactual_credit_updates = 7
    actor.counterfactual_credit_last_global_step = 9
    with torch.no_grad():
        actor.counterfactual_credit_head.output_layer.weight.fill_(1.0)

    actor.load_counterfactual_credit_state_dict(
        {
            "format_version": 1,
            "architecture": {
                "type": "h_pre_delta_two_layer_mlp",
                "hidden_size": 4,
                "projection_size": 5,
            },
        }
    )

    assert actor.counterfactual_credit_updates == 0
    assert actor.counterfactual_credit_last_global_step is None
    assert actor.counterfactual_credit_head.output_layer.weight.any()
    assert actor.counterfactual_credit_head.output_layer.weight.std() < 0.01
    assert not actor.counterfactual_credit_head.output_layer.bias.any()
