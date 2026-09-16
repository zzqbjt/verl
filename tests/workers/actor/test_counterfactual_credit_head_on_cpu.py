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

from copy import deepcopy
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


def _make_actor(
    use_remove_padding: bool,
    *,
    use_dynamic_bsz: bool = False,
) -> DataParallelPPOActor:
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
    # Exact probe metric surface of Qwen3-1.7B-S2D-Credit-0.3, without
    # duplicate phase metrics or any additional post-update evaluation.
    expected_keys = {
        "credit_predictions",
        "credit_head_direction_agreement",
        "credit_head_direction_majority_baseline",
        "credit_head_direction_excess",
    }
    expected_keys.update(
        f"credit_head_value_{metric}{source}"
        for metric in ("bce", "mae")
        for source in ("", "_mc", "_start", "_terminal")
    )
    assert set(outputs) == expected_keys
    expected_credit = torch.tensor([[0.5, 0.0, 0.25], [0.0, -0.5, 0.0]])
    torch.testing.assert_close(outputs["credit_predictions"], expected_credit)
    assert torch.isfinite(outputs["credit_head_value_bce"]).all()
    for source_name in ("mc", "start", "terminal"):
        assert torch.isfinite(outputs[f"credit_head_value_bce_{source_name}"]).all()
        assert torch.isfinite(outputs[f"credit_head_value_mae_{source_name}"]).all()
    torch.testing.assert_close(
        outputs["credit_head_value_bce"],
        0.6 * outputs["credit_head_value_bce_mc"]
        + 0.2 * outputs["credit_head_value_bce_start"]
        + 0.2 * outputs["credit_head_value_bce_terminal"],
    )
    torch.testing.assert_close(
        outputs["credit_head_value_mae"],
        0.6 * outputs["credit_head_value_mae_mc"]
        + 0.2 * outputs["credit_head_value_mae_start"]
        + 0.2 * outputs["credit_head_value_mae_terminal"],
    )
    assert "credit_head_difference_smooth_l1" not in outputs
    assert "credit_head_total_loss" not in outputs
    assert torch.all((outputs["credit_head_value_mae"] >= 0) & (outputs["credit_head_value_mae"] <= 1))
    assert torch.all(
        (outputs["credit_head_direction_agreement"] >= 0) & (outputs["credit_head_direction_agreement"] <= 1)
    )
    assert torch.all(
        (outputs["credit_head_direction_majority_baseline"] >= 0.5)
        & (outputs["credit_head_direction_majority_baseline"] <= 1)
    )
    assert torch.all((outputs["credit_head_direction_excess"] >= -1) & (outputs["credit_head_direction_excess"] <= 0.5))
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
    assert actor.counterfactual_credit_updates == 1


@pytest.mark.parametrize("predict_all_steps", [False, True])
@pytest.mark.parametrize("source_weights", [(0.6, 0.2, 0.2), (0.6, 0.0, 0.2), (0.6, 0.2, 0.0), (1.0, 0.0, 0.0)])
def test_probe_skips_disabled_sources_and_matches_bce_only(predict_all_steps, source_weights):
    optimizer_steps = 1
    actor = _make_actor(False)
    reference_head, reference_optimizer = deepcopy(
        (actor.counterfactual_credit_head, actor.counterfactual_credit_optimizer)
    )
    # These are the MC-Q, prompt, and terminal boundaries in _make_data.
    source_hidden = (
        torch.full((1, 4), 0.3),
        torch.full((1, 4), 0.2),
        torch.tensor([[0.5] * 4, [0.8] * 4]),
    )
    source_targets = (torch.tensor([0.75]), torch.tensor([0.25]), torch.tensor([1.0, 0.0]))
    for step in range(optimizer_steps):
        bce = sum(
            weight
            / sum(source_weights)
            * torch.nn.functional.binary_cross_entropy(reference_head(hidden).squeeze(-1), targets)
            for weight, hidden, targets in zip(source_weights, source_hidden, source_targets, strict=True)
            if weight > 0
        )
        if step == 0:
            expected_pre_update_bce = bce.detach().clone()
        reference_optimizer.zero_grad(set_to_none=True)
        bce.backward()
        reference_optimizer.step()

    data = _make_data()
    data.meta_info["predict_all_credit_steps"] = predict_all_steps
    data.meta_info["credit_value_loss_weights"] = source_weights
    forward_grad_modes = []
    diagnostic_inputs = []

    def check_forward(_module, args):
        forward_grad_modes.append(torch.is_grad_enabled())
        if not torch.is_grad_enabled():
            diagnostic_inputs.append(args[0].detach().clone())
            return
        # Disabled boundaries may be predicted for diagnostics, but not trained.
        forbidden_values = []
        if source_weights[1] == 0:
            forbidden_values.extend((0.2, 0.6))
        if source_weights[2] == 0:
            forbidden_values.extend((0.5, 0.8))
        for value in forbidden_values:
            assert not torch.isclose(args[0][:, 0], torch.tensor(value)).any()

    hook = actor.counterfactual_credit_head.register_forward_pre_hook(check_forward)
    try:
        with (
            patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
            patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
            patch(
                "torch.nn.functional.smooth_l1_loss",
                side_effect=AssertionError("Probe training must only compute value BCE."),
            ),
        ):
            outputs = actor.compute_counterfactual_credit(data)
    finally:
        hook.remove()

    # Only enabled BCE sources build graphs; direction/policy prediction remains no-grad.
    assert sum(forward_grad_modes) == sum(weight > 0 for weight in source_weights) * optimizer_steps
    predicted_tokens = torch.cat(diagnostic_inputs)[:, 0]
    for boundary in (0.2, 0.6, 0.8):
        assert torch.isclose(predicted_tokens, torch.tensor(boundary)).any()
    assert actor.counterfactual_credit_updates == optimizer_steps
    torch.testing.assert_close(outputs["credit_head_value_bce"], expected_pre_update_bce.expand(2))
    assert "credit_head_total_loss" not in outputs
    assert "credit_head_difference_smooth_l1" not in outputs
    for weight, source_name in zip(source_weights, ("mc", "start", "terminal"), strict=True):
        for metric in ("bce", "mae"):
            source_metric = outputs[f"credit_head_value_{metric}_{source_name}"]
            assert source_metric.isfinite().all() if weight > 0 else source_metric.isnan().all()
    assert outputs["credit_head_direction_agreement"].isfinite().all()
    assert outputs["credit_head_direction_excess"].isfinite().all()
    expected_credit = torch.tensor([[0.5, 0.0, 0.25], [0.0, -0.5, 0.0]])
    if not predict_all_steps:
        expected_credit.zero_()
    torch.testing.assert_close(outputs["credit_predictions"], expected_credit)
    for actual, expected in zip(
        actor.counterfactual_credit_head.parameters(), reference_head.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


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


def test_mc_only_without_interior_labels_only_predicts_for_direction_and_skips_optimizer():
    actor = _make_actor(False)
    # The second response has one step, with both boundary values known.
    data = _make_data()[1:]
    data.meta_info["credit_value_loss_weights"] = (1.0, 0.0, 0.0)

    def diagnostic_forward(hidden):
        assert not torch.is_grad_enabled()
        torch.testing.assert_close(hidden, torch.tensor([[0.6] * 4, [0.8] * 4]))
        return hidden.new_full((hidden.shape[0], 1), 0.5)

    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
        patch.object(
            actor.counterfactual_credit_head,
            "forward",
            side_effect=diagnostic_forward,
        ) as head_forward,
        patch.object(actor.counterfactual_credit_optimizer, "step") as optimizer_step,
    ):
        outputs = actor.compute_counterfactual_credit(data)

    optimizer_step.assert_not_called()
    assert head_forward.call_count == 1  # Only pre-update diagnostics; no trainable labels.
    assert actor.counterfactual_credit_updates == 0
    torch.testing.assert_close(outputs["credit_predictions"], torch.tensor([[0.0, -0.5, 0.0]]))
    torch.testing.assert_close(outputs["credit_head_direction_agreement"], torch.zeros(1))
    for source_name in ("mc", "start", "terminal"):
        assert outputs[f"credit_head_value_bce_{source_name}"].isnan().all()
        assert outputs[f"credit_head_value_mae_{source_name}"].isnan().all()


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


@pytest.mark.parametrize("source_weights", [(0.6, 0.2, 0.2), (1.0, 0.0, 0.0)])
def test_dynamic_credit_micro_batches_restore_original_response_order(source_weights):
    actor = _make_actor(True, use_dynamic_bsz=True)
    with torch.no_grad():
        actor.counterfactual_credit_head.output_layer.bias.fill_(0.25)
    data = _make_data()
    data.meta_info["use_dynamic_bsz"] = True
    data.meta_info["max_token_len"] = 5
    data.meta_info["credit_value_loss_weights"] = source_weights
    data.batch["step_end_mask"][0].fill_(True)
    with torch.no_grad():
        old_value = actor.counterfactual_credit_head(torch.full((1, 4), 0.4)).item()
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
    ):
        outputs = actor.compute_counterfactual_credit(data)

    expected_credit = torch.tensor([[0.5, old_value - 0.75, 1.0 - old_value], [0.0, -0.5, 0.0]])
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
    torch.testing.assert_close(outputs["credit_head_direction_majority_baseline"], torch.full((2,), 0.8))
    torch.testing.assert_close(outputs["credit_head_direction_excess"], torch.zeros(2))


@pytest.mark.parametrize("source_weights", [(0.6, 0.2, 0.2), (1.0, 0.0, 0.0)])
@pytest.mark.parametrize("predict_all_steps", [False, True])
@pytest.mark.parametrize("predicted_value", [0.1, 0.5])
def test_direction_uses_raw_predictions_without_boundary_or_mc_substitution(
    source_weights, predict_all_steps, predicted_value
):
    actor = _make_actor(False)
    with torch.no_grad():
        for parameter in actor.counterfactual_credit_head.parameters():
            parameter.zero_()
        actor.counterfactual_credit_head.output_layer.bias.fill_(torch.logit(torch.tensor(predicted_value)))
    data = _make_data()[:1]
    data.meta_info["credit_value_loss_weights"] = source_weights
    data.meta_info["predict_all_credit_steps"] = predict_all_steps
    data.batch["step_end_mask"].fill_(True)
    data.batch["credit_anchor_mask"].fill_(True)
    data.batch["credit_anchor_q_targets"] = torch.tensor([[0.75, 0.25, 1.0]])
    data.batch["credit_anchor_v_targets"] = torch.tensor([[0.25, 0.75, 0.25]])
    data.batch["credit_anchor_targets"] = torch.tensor([[0.5, -0.5, 0.75]])
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
    ):
        outputs = actor.compute_counterfactual_credit(data)

    # All predicted values are identical, so every raw difference is zero.
    # Neither boundary nor interior MC labels may improve this diagnostic.
    torch.testing.assert_close(outputs["credit_head_direction_agreement"], torch.zeros(1))
    torch.testing.assert_close(outputs["credit_head_direction_majority_baseline"], torch.tensor([5 / 7]))
    torch.testing.assert_close(outputs["credit_head_direction_excess"], torch.tensor([-5 / 7]))
    expected_credit = data.batch["credit_anchor_targets"] if predict_all_steps else torch.zeros((1, 3))
    torch.testing.assert_close(outputs["credit_predictions"], expected_credit)


@pytest.mark.parametrize("source_weights", [(0.6, 0.2, 0.2), (1.0, 0.0, 0.0)])
@pytest.mark.parametrize("use_remove_padding", [False, True])
def test_policy_and_metrics_use_pre_update_probe_with_one_optimizer_step(source_weights, use_remove_padding):
    optimizer_steps = 1
    actor = _make_actor(use_remove_padding)
    with torch.no_grad():
        for parameter in actor.counterfactual_credit_head.parameters():
            parameter.zero_()
        actor.counterfactual_credit_head.output_layer.bias.fill_(0.25)

    optimizer_calls = 0

    def replace_head_during_optimizer_step():
        nonlocal optimizer_calls
        optimizer_calls += 1
        with torch.no_grad():
            actor.counterfactual_credit_head.input_layer.weight[0, 0] = 1.0
            actor.counterfactual_credit_head.output_layer.weight[0, 0] = optimizer_calls
            actor.counterfactual_credit_head.output_layer.bias.fill_(0.75)

    data = _make_data()
    data.meta_info["credit_value_loss_weights"] = source_weights
    # Three steps: MC fixes boundary 0.3, while boundary 0.4 needs the probe.
    data.batch["step_end_mask"][0].fill_(True)
    prediction_calls = []

    def record_prediction(_module, args):
        if not torch.is_grad_enabled():
            prediction_calls.append((actor.counterfactual_credit_updates, args[0].detach().clone()))

    hook = actor.counterfactual_credit_head.register_forward_pre_hook(record_prediction)
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
        patch.object(
            actor.counterfactual_credit_optimizer,
            "step",
            side_effect=replace_head_during_optimizer_step,
        ),
        patch.object(actor, "_forward_micro_batch", wraps=actor._forward_micro_batch) as backbone_forward,
    ):
        try:
            outputs = actor.compute_counterfactual_credit(data)
        finally:
            hook.remove()

    old_value = torch.sigmoid(torch.tensor(0.25))
    new_value = torch.sigmoid(0.75 + optimizer_steps * torch.nn.functional.silu(torch.tensor(0.4)))
    assert not torch.isclose(old_value, new_value)
    policy_value = old_value
    expected = torch.tensor([[0.5, policy_value - 0.75, 1.0 - policy_value], [0.0, -0.5, 0.0]])
    torch.testing.assert_close(outputs["credit_predictions"], expected)
    # The old constant predictor has zero credit everywhere. The updated head
    # is increasing, so the raw post-update direction agreement would be 0.5.
    torch.testing.assert_close(outputs["credit_head_direction_agreement"], torch.zeros(2))
    torch.testing.assert_close(outputs["credit_head_direction_majority_baseline"], torch.full((2,), 0.5))
    torch.testing.assert_close(outputs["credit_head_direction_excess"], torch.full((2,), -0.5))
    assert not any(key.endswith("_pre_update") for key in outputs)
    assert "credit_head_direction_majority_baseline_post_update" not in outputs
    assert not any(key.endswith("_post_update") for key in outputs)
    expected_bce = torch.tensor(0.0)
    expected_mae = torch.tensor(0.0)
    for weight, source, targets in zip(
        source_weights,
        ("mc", "start", "terminal"),
        (torch.tensor([0.75]), torch.tensor([0.25]), torch.tensor([1.0, 0.0])),
        strict=True,
    ):
        if weight == 0:
            assert outputs[f"credit_head_value_bce_{source}"].isnan().all()
            assert outputs[f"credit_head_value_mae_{source}"].isnan().all()
            continue
        old_predictions = old_value.expand_as(targets)
        bce = torch.nn.functional.binary_cross_entropy(old_predictions, targets)
        mae = (old_predictions - targets).abs().mean()
        torch.testing.assert_close(outputs[f"credit_head_value_bce_{source}"], bce.expand(2))
        torch.testing.assert_close(outputs[f"credit_head_value_mae_{source}"], mae.expand(2))
        expected_bce += weight * bce
        expected_mae += weight * mae
    torch.testing.assert_close(outputs["credit_head_value_bce"], expected_bce.expand(2))
    torch.testing.assert_close(outputs["credit_head_value_mae"], expected_mae.expand(2))
    torch.testing.assert_close(actor.counterfactual_credit_head.output_layer.bias, torch.tensor([0.75]))
    assert optimizer_calls == actor.counterfactual_credit_updates == optimizer_steps
    assert backbone_forward.call_count == 1
    assert all(parameter.grad is None for parameter in actor.actor_module.parameters())
    assert [updates for updates, _ in prediction_calls] == [0]
    dense_call = prediction_calls[0][1][:, 0]
    torch.testing.assert_close(dense_call, torch.tensor([0.2, 0.3, 0.4, 0.6, 0.8]))


def test_repeated_global_step_does_not_update_probe_twice():
    actor = _make_actor(False)
    data = _make_data()
    data.batch["step_end_mask"][0].fill_(True)
    with torch.no_grad():
        old_value = actor.counterfactual_credit_head(torch.full((1, 4), 0.4)).item()
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
        patch.object(
            actor.counterfactual_credit_optimizer, "step", wraps=actor.counterfactual_credit_optimizer.step
        ) as optimizer_step,
    ):
        first_outputs = actor.compute_counterfactual_credit(data)
        with torch.no_grad():
            updated_value = actor.counterfactual_credit_head(torch.full((1, 4), 0.4)).item()
        second_outputs = actor.compute_counterfactual_credit(data)

    assert actor.counterfactual_credit_updates == optimizer_step.call_count == 1
    assert not any(key.endswith(("_pre_update", "_post_update")) for key in second_outputs)
    torch.testing.assert_close(
        first_outputs["credit_predictions"],
        torch.tensor([[0.5, old_value - 0.75, 1.0 - old_value], [0.0, -0.5, 0.0]]),
    )
    torch.testing.assert_close(
        second_outputs["credit_predictions"],
        torch.tensor([[0.5, updated_value - 0.75, 1.0 - updated_value], [0.0, -0.5, 0.0]]),
    )


def test_zero_lambda_step_predicts_only_mc_anchors_for_probe_update():
    actor = _make_actor(False)
    data = _make_data()
    data.meta_info["predict_all_credit_steps"] = False
    data.batch["step_end_mask"][0].fill_(True)
    prediction_calls = []

    def record_prediction(_module, args):
        if not torch.is_grad_enabled():
            prediction_calls.append((actor.counterfactual_credit_updates, args[0].detach().clone()))

    hook = actor.counterfactual_credit_head.register_forward_pre_hook(record_prediction)
    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=_cpu_logprobs),
    ):
        try:
            outputs = actor.compute_counterfactual_credit(data)
        finally:
            hook.remove()

    assert len(prediction_calls) == 1  # Only pre-update anchor diagnostics; no post-update pass.
    assert prediction_calls[0][0] == 0
    torch.testing.assert_close(prediction_calls[0][1][:, 0], torch.tensor([0.2, 0.3, 0.6, 0.8]))
    assert not any(torch.isclose(hidden[:, 0], torch.tensor(0.4)).any() for _, hidden in prediction_calls)
    torch.testing.assert_close(outputs["credit_predictions"], torch.zeros_like(data.batch["step_end_mask"].float()))
    assert actor.counterfactual_credit_updates == 1
    assert torch.isfinite(outputs["credit_head_value_bce"]).all()
    assert "credit_head_difference_smooth_l1" not in outputs
    assert "credit_head_total_loss" not in outputs
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

    assert restored.counterfactual_credit_updates == 1
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
