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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.step_split import build_step_start_mask
from verl.workers.actor.dp_actor import DataParallelPPOActor, StepValueProbe
from verl.workers.config import FSDPActorConfig, StepValueProbeConfig


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vision_config=None)
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            values = torch.arange(vocab_size, dtype=torch.float32).unsqueeze(-1)
            self.embed_tokens.weight.copy_(values.repeat(1, hidden_size) / 10.0)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, **kwargs):
        assert "output_hidden_states" not in kwargs
        hidden = self.embed_tokens(input_ids)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)


def make_actor(
    *,
    use_remove_padding: bool,
    use_dynamic_bsz: bool = False,
    probe_hidden_dim: int = 3,
    probe_enabled: bool = True,
) -> DataParallelPPOActor:
    config = FSDPActorConfig(
        strategy="fsdp",
        rollout_n=1,
        use_dynamic_bsz=use_dynamic_bsz,
        ppo_micro_batch_size_per_gpu=2,
        use_remove_padding=use_remove_padding,
        use_torch_compile=False,
        ulysses_sequence_parallel_size=1,
        step_value_probe=StepValueProbeConfig(
            enabled=probe_enabled,
            hidden_dim=probe_hidden_dim,
            lr=1e-2,
            weight_decay=0.0,
            warmup_epochs=2,
            update_epochs=1,
            warmup_updates=1,
        ),
    )
    model = TinyCausalLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    with patch("torch.distributed.get_rank", return_value=0):
        actor = DataParallelPPOActor(config, model, optimizer)
    actor.device_name = "cpu"
    if probe_enabled:
        with torch.no_grad():
            actor.step_value_probe.input_layer.weight.zero_()
            actor.step_value_probe.input_layer.bias.zero_()
            actor.step_value_probe.input_layer.weight[0, 0] = 1.0
            actor.step_value_probe.output_layer.weight.zero_()
            actor.step_value_probe.output_layer.weight[0, 0] = 1.0
            actor.step_value_probe.output_layer.bias.zero_()
        actor._step_value_probe_needs_broadcast = False
    return actor


def compute_step_values(actor: DataParallelPPOActor, data: DataProto):
    def cpu_logprobs(logits, labels, **_):
        return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=cpu_logprobs),
    ):
        return actor.compute_log_prob(data, compute_step_value_probe=True)


def compute_similarity_embeddings(actor: DataParallelPPOActor, data: DataProto):
    def cpu_logprobs(logits, labels, **_):
        return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    with (
        patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=cpu_logprobs),
    ):
        return actor.compute_log_prob(data, compute_similarity_step_embeddings=True)


def make_data(global_step: int) -> DataProto:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [0, 6, 7, 8, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    responses = input_ids[:, -3:]
    step_end_mask = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
        ]
    )
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": attention_mask.cumsum(dim=-1) - 1,
            "responses": responses,
            "step_end_mask": step_end_mask,
            "step_value_targets": torch.tensor([1.0, 0.0]),
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
        },
    )


def make_dynamic_data(global_step: int) -> DataProto:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7],
            [0, 8, 9, 10, 0, 0, 0],
            [11, 12, 13, 14, 15, 16, 0],
            [0, 17, 18, 19, 20, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    responses = input_ids[:, -4:]
    step_end_mask = torch.tensor(
        [
            [True, False, False, True],
            [True, False, False, False],
            [False, True, True, False],
            [True, True, False, False],
        ]
    )
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": attention_mask.cumsum(dim=-1) - 1,
        "responses": responses,
        "step_end_mask": step_end_mask,
        "step_value_targets": targets,
    }
    batch = TensorDict(tensors, batch_size=[4])
    return DataProto(
        batch=batch,
        meta_info={
            "micro_batch_size": 4,
            "max_token_len": 7,
            "temperature": 1.0,
            "use_dynamic_bsz": True,
            "global_steps": global_step,
        },
    )


def add_delayed_audit_contract(
    data: DataProto,
    *,
    update_mask: torch.Tensor,
    audit_mask: torch.Tensor,
    row_ids: torch.Tensor,
) -> DataProto:
    data.batch["step_value_probe_update_mask"] = update_mask
    data.batch["step_value_prompt_center_audit_mask"] = audit_mask
    data.batch["step_value_forward_row_id"] = row_ids
    return data


def run_two_rank_delayed_audit_worker(rank: int, world_size: int, init_method: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        actor = make_actor(use_remove_padding=True, use_dynamic_bsz=True)
        global_data = make_dynamic_data(global_step=1)
        local_indices = torch.arange(rank * 2, rank * 2 + 2)
        local_batch = TensorDict(
            {key: value.index_select(0, local_indices) for key, value in global_data.batch.items()},
            batch_size=[2],
        )
        data = DataProto(
            batch=local_batch,
            meta_info=dict(global_data.meta_info),
        )
        update_mask = torch.tensor([rank == 0, rank == 0])
        audit_mask = torch.tensor([rank == 1, False])
        data = add_delayed_audit_contract(
            data,
            update_mask=update_mask,
            audit_mask=audit_mask,
            row_ids=local_indices,
        )

        outputs = compute_step_values(actor, data)

        assert int(update_mask.sum().item()) == (2 if rank == 0 else 0)
        assert actor.step_value_probe_updates == 1
        assert actor.step_value_probe_warmup_completed_at == 1
        assert actor.step_value_probe_last_global_step == 1
        torch.testing.assert_close(outputs["step_value_forward_row_id"], local_indices)
        expected_audit_counts = [2, 0] if rank == 1 else [0, 0]
        expected_audit_readiness = [True, False] if rank == 1 else [False, False]
        assert outputs["step_value_audit_endpoint_count"].tolist() == expected_audit_counts
        assert outputs["step_value_audit_ready_next"].tolist() == expected_audit_readiness

        audit_step_mask = global_data.batch["step_end_mask"][2]
        audit_endpoint_ids = global_data.batch["responses"][2][audit_step_mask]
        audit_hidden = audit_endpoint_ids.float().unsqueeze(-1).repeat(1, 4) / 10.0
        with torch.no_grad():
            expected_post_update_mean = actor.step_value_probe(audit_hidden).squeeze(-1).mean()
        if rank == 1:
            torch.testing.assert_close(
                outputs["step_value_audit_trajectory_logit_mean"][0],
                expected_post_update_mean,
            )
            assert not torch.isclose(
                outputs["step_value_audit_trajectory_logit_mean"][0],
                outputs["step_value_trajectory_logit_mean"][0],
            )

        probe_vector = torch.nn.utils.parameters_to_vector(actor.step_value_probe.parameters()).detach()
        gathered_probe_vectors = [torch.empty_like(probe_vector) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_probe_vectors, probe_vector)
        for other_probe_vector in gathered_probe_vectors[1:]:
            torch.testing.assert_close(other_probe_vector, gathered_probe_vectors[0], rtol=0, atol=0)

        gathered_audit_means = [torch.empty_like(expected_post_update_mean) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_audit_means, expected_post_update_mean)
        for other_audit_mean in gathered_audit_means[1:]:
            torch.testing.assert_close(other_audit_mean, gathered_audit_means[0], rtol=0, atol=0)

        gathered_row_ids = [torch.empty_like(local_indices) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_row_ids, outputs["step_value_forward_row_id"])
        torch.testing.assert_close(torch.cat(gathered_row_ids), torch.arange(4), rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("use_remove_padding", [False, True])
def test_probe_uses_the_step_tail_token_and_warms_up_before_becoming_ready(use_remove_padding: bool):
    actor = make_actor(use_remove_padding=use_remove_padding)

    assert actor.step_value_probe.input_layer.in_features == 4
    assert actor.step_value_probe.input_layer.out_features == 3
    assert isinstance(actor.step_value_probe.activation, torch.nn.SiLU)
    assert actor.step_value_probe.output_layer.in_features == 3
    assert actor.step_value_probe.output_layer.out_features == 1

    first_outputs = compute_step_values(actor, make_data(global_step=1))

    assert set(first_outputs) == {
        "log_probs",
        "step_values",
        "step_value_trajectory_logit_mean",
        "step_value_ready",
        "step_value_probe_loss",
        "step_value_probe_grad_norm",
    }
    expected = torch.zeros(2, 3)
    expected[0, 0] = torch.sigmoid(torch.nn.functional.silu(torch.tensor(0.3)))
    expected[0, 2] = 1.0
    expected[1, 1] = 0.0
    torch.testing.assert_close(first_outputs["step_values"], expected)
    expected_logit_means = torch.tensor(
        [
            torch.stack(
                (
                    torch.nn.functional.silu(torch.tensor(0.3)),
                    torch.nn.functional.silu(torch.tensor(0.5)),
                )
            ).mean(),
            torch.nn.functional.silu(torch.tensor(0.8)),
        ]
    )
    torch.testing.assert_close(first_outputs["step_value_trajectory_logit_mean"], expected_logit_means)
    assert first_outputs["step_value_ready"].tolist() == [False, False]
    assert torch.isfinite(first_outputs["step_value_probe_loss"]).all()
    assert (first_outputs["step_value_probe_grad_norm"] > 0).all()

    second_outputs = compute_step_values(actor, make_data(global_step=2))

    assert second_outputs["step_value_ready"].tolist() == [True, True]


def test_probe_checkpoint_round_trip_and_missing_checkpoint_restarts_warmup():
    actor = make_actor(use_remove_padding=False)
    compute_step_values(actor, make_data(global_step=1))
    saved_state = actor.step_value_probe_state_dict()

    assert saved_state["format_version"] == 2
    assert saved_state["architecture"] == {
        "type": "two_layer_mlp",
        "input_dim": 4,
        "hidden_dim": 3,
    }

    restored = make_actor(use_remove_padding=False)
    restored.load_step_value_probe_state_dict(saved_state)

    assert restored.step_value_probe_updates == 1
    assert restored.step_value_probe_warmup_completed_at == 1
    for expected, actual in zip(
        actor.step_value_probe.parameters(),
        restored.step_value_probe.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)

    restored.load_step_value_probe_state_dict(None)
    assert restored.step_value_probe_updates == 0
    assert restored.step_value_probe_warmup_completed_at is None
    assert restored.step_value_probe_last_global_step is None


def test_loading_v2_probe_checkpoint_rejects_architecture_mismatch():
    actor = make_actor(use_remove_padding=False, probe_hidden_dim=3)
    saved_state = actor.step_value_probe_state_dict()
    saved_state["architecture"] = {
        "type": "two_layer_mlp",
        "input_dim": 4,
        "hidden_dim": 5,
    }

    with pytest.raises(RuntimeError, match="architecture does not match"):
        actor.load_step_value_probe_state_dict(saved_state)


def test_loading_legacy_linear_probe_checkpoint_is_not_supported():
    actor = make_actor(use_remove_padding=False)

    with pytest.raises(RuntimeError, match="Unsupported step-value probe checkpoint format"):
        actor.load_step_value_probe_state_dict({"format_version": 1})


def test_dynamic_micro_batches_restore_outputs_and_keep_training_examples_aligned():
    actor = make_actor(use_remove_padding=True, use_dynamic_bsz=True)
    data = make_dynamic_data(global_step=1)
    initial_probe_state = {
        name: tensor.detach().clone() for name, tensor in actor.step_value_probe.state_dict().items()
    }

    outputs = compute_step_values(actor, data)

    step_end_mask = data.batch["step_end_mask"]
    targets = data.batch["step_value_targets"]
    endpoint_token_ids = data.batch["responses"][step_end_mask]
    endpoint_logits = torch.nn.functional.silu(endpoint_token_ids.float() / 10.0)
    expected_values = torch.zeros_like(outputs["step_values"])
    expected_values[step_end_mask] = torch.sigmoid(endpoint_logits)
    endpoint_positions = torch.arange(step_end_mask.shape[1]).expand_as(step_end_mask)
    terminal_positions = endpoint_positions.masked_fill(~step_end_mask, -1).max(dim=-1).values
    expected_values[torch.arange(step_end_mask.shape[0]), terminal_positions] = targets
    torch.testing.assert_close(outputs["step_values"], expected_values)

    step_counts = step_end_mask.sum(dim=-1)
    sample_indices = torch.arange(4).repeat_interleave(step_counts)
    endpoint_targets = targets.index_select(0, sample_indices)
    endpoint_weights = step_counts.float().reciprocal().index_select(0, sample_indices)
    expected_logit_means = torch.zeros(4)
    expected_logit_means.scatter_add_(0, sample_indices, endpoint_logits * endpoint_weights)
    torch.testing.assert_close(outputs["step_value_trajectory_logit_mean"], expected_logit_means)
    endpoint_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        endpoint_logits,
        endpoint_targets,
        reduction="none",
    )
    expected_trajectory_losses = torch.zeros(4)
    expected_trajectory_losses.scatter_add_(0, sample_indices, endpoint_losses * endpoint_weights)
    torch.testing.assert_close(outputs["step_value_probe_loss"], expected_trajectory_losses)

    reference_probe = StepValueProbe(input_dim=4, hidden_dim=3)
    reference_probe.load_state_dict(initial_probe_state)
    reference_optimizer = torch.optim.AdamW(reference_probe.parameters(), lr=1e-2, weight_decay=0.0)
    endpoint_hidden = endpoint_token_ids.float().unsqueeze(-1).repeat(1, 4) / 10.0
    for _ in range(2):
        reference_optimizer.zero_grad(set_to_none=True)
        logits = reference_probe(endpoint_hidden).squeeze(-1)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, endpoint_targets, reduction="none")
        (losses * endpoint_weights).sum().div(4).backward()
        reference_optimizer.step()
    expected_grad_norm = torch.sqrt(
        sum(parameter.grad.detach().float().square().sum() for parameter in reference_probe.parameters())
    )
    torch.testing.assert_close(outputs["step_value_probe_grad_norm"], expected_grad_norm.expand(4))
    for expected, actual in zip(reference_probe.parameters(), actor.step_value_probe.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_delayed_audit_reuses_dynamic_forward_without_changing_main_probe_update():
    combined_actor = make_actor(use_remove_padding=True, use_dynamic_bsz=True)
    main_only_actor = make_actor(use_remove_padding=True, use_dynamic_bsz=False)
    combined_data = add_delayed_audit_contract(
        make_dynamic_data(global_step=1),
        update_mask=torch.tensor([True, False, True, False]),
        audit_mask=torch.tensor([False, True, True, False]),
        row_ids=torch.tensor([901, 17, 450, 63]),
    )

    main_indices = torch.tensor([0, 2])
    main_source = make_dynamic_data(global_step=1)
    main_batch = TensorDict(
        {key: value.index_select(0, main_indices) for key, value in main_source.batch.items()},
        batch_size=[2],
    )
    main_data = DataProto(
        batch=main_batch,
        meta_info={
            "micro_batch_size": 2,
            "temperature": 1.0,
            "use_dynamic_bsz": False,
            "global_steps": 1,
        },
    )

    combined_outputs = compute_step_values(combined_actor, combined_data)
    compute_step_values(main_only_actor, main_data)

    assert combined_actor.step_value_probe_updates == main_only_actor.step_value_probe_updates == 1
    assert combined_actor.step_value_probe_warmup_completed_at == 1
    assert combined_actor.step_value_probe_last_global_step == 1
    for combined_parameter, main_parameter in zip(
        combined_actor.step_value_probe.parameters(),
        main_only_actor.step_value_probe.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(combined_parameter, main_parameter)

    assert combined_outputs["step_value_forward_row_id"].tolist() == [901, 17, 450, 63]
    assert combined_outputs["step_value_audit_endpoint_count"].tolist() == [0, 1, 2, 0]
    assert combined_outputs["step_value_audit_ready_next"].tolist() == [False, True, True, False]
    assert combined_outputs["step_value_ready"].tolist() == [False, False, False, False]

    step_end_mask = combined_data.batch["step_end_mask"]
    audit_mask = combined_data.batch["step_value_prompt_center_audit_mask"]
    endpoint_row_indices = torch.arange(4).repeat_interleave(step_end_mask.sum(dim=-1))
    endpoint_token_ids = combined_data.batch["responses"][step_end_mask]
    endpoint_hidden = endpoint_token_ids.float().unsqueeze(-1).repeat(1, 4) / 10.0
    with torch.no_grad():
        post_update_logits = combined_actor.step_value_probe(endpoint_hidden).squeeze(-1)
    expected_audit_means = torch.zeros(4)
    expected_audit_counts = step_end_mask.sum(dim=-1) * audit_mask
    endpoint_audit_mask = audit_mask.index_select(0, endpoint_row_indices)
    expected_audit_means.scatter_add_(
        0,
        endpoint_row_indices[endpoint_audit_mask],
        post_update_logits[endpoint_audit_mask],
    )
    expected_audit_means /= expected_audit_counts.clamp_min(1)
    torch.testing.assert_close(
        combined_outputs["step_value_audit_trajectory_logit_mean"],
        expected_audit_means,
    )
    assert not torch.equal(
        combined_outputs["step_value_audit_trajectory_logit_mean"][audit_mask],
        combined_outputs["step_value_trajectory_logit_mean"][audit_mask],
    )


def test_strict_disjoint_main_audit_and_neutral_rows_survive_dynamic_reordering():
    combined_actor = make_actor(use_remove_padding=True, use_dynamic_bsz=True)
    main_only_actor = make_actor(use_remove_padding=True, use_dynamic_bsz=False)
    combined_data = add_delayed_audit_contract(
        make_dynamic_data(global_step=1),
        update_mask=torch.tensor([True, True, False, False]),
        audit_mask=torch.tensor([False, False, True, False]),
        row_ids=torch.tensor([7003, 11, 902, 44]),
    )
    assert not torch.any(
        combined_data.batch["step_value_probe_update_mask"]
        & combined_data.batch["step_value_prompt_center_audit_mask"]
    )

    main_indices = torch.tensor([0, 1])
    main_source = make_dynamic_data(global_step=1)
    main_batch = TensorDict(
        {key: value.index_select(0, main_indices) for key, value in main_source.batch.items()},
        batch_size=[2],
    )
    main_data = DataProto(
        batch=main_batch,
        meta_info={
            "micro_batch_size": 2,
            "temperature": 1.0,
            "use_dynamic_bsz": False,
            "global_steps": 1,
        },
    )

    combined_outputs = compute_step_values(combined_actor, combined_data)
    compute_step_values(main_only_actor, main_data)

    for combined_parameter, main_parameter in zip(
        combined_actor.step_value_probe.parameters(),
        main_only_actor.step_value_probe.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(combined_parameter, main_parameter)
    assert combined_actor.step_value_probe_updates == main_only_actor.step_value_probe_updates == 1
    assert combined_outputs["step_value_forward_row_id"].tolist() == [7003, 11, 902, 44]
    assert combined_outputs["step_value_audit_endpoint_count"].tolist() == [0, 0, 2, 0]
    assert combined_outputs["step_value_audit_ready_next"].tolist() == [False, False, True, False]
    torch.testing.assert_close(
        combined_outputs["step_value_audit_trajectory_logit_mean"][[0, 1, 3]],
        torch.zeros(3),
        rtol=0,
        atol=0,
    )

    audit_step_mask = combined_data.batch["step_end_mask"][2]
    audit_endpoint_ids = combined_data.batch["responses"][2][audit_step_mask]
    audit_hidden = audit_endpoint_ids.float().unsqueeze(-1).repeat(1, 4) / 10.0
    with torch.no_grad():
        expected_audit_mean = combined_actor.step_value_probe(audit_hidden).squeeze(-1).mean()
    torch.testing.assert_close(
        combined_outputs["step_value_audit_trajectory_logit_mean"][2],
        expected_audit_mean,
    )


def test_audit_only_rows_do_not_advance_or_optimize_probe():
    actor = make_actor(use_remove_padding=False)
    data = add_delayed_audit_contract(
        make_data(global_step=1),
        update_mask=torch.tensor([False, False]),
        audit_mask=torch.tensor([True, True]),
        row_ids=torch.tensor([40, 12]),
    )
    parameters_before = [parameter.detach().clone() for parameter in actor.step_value_probe.parameters()]

    outputs = compute_step_values(actor, data)

    assert actor.step_value_probe_updates == 0
    assert actor.step_value_probe_warmup_completed_at is None
    assert actor.step_value_probe_last_global_step is None
    assert actor.step_value_probe_optimizer.state == {}
    torch.testing.assert_close(outputs["step_value_probe_grad_norm"], torch.zeros(2))
    assert outputs["step_value_audit_endpoint_count"].tolist() == [2, 1]
    assert outputs["step_value_audit_ready_next"].tolist() == [False, False]
    torch.testing.assert_close(
        outputs["step_value_audit_trajectory_logit_mean"],
        outputs["step_value_trajectory_logit_mean"],
    )
    for expected, actual in zip(parameters_before, actor.step_value_probe.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rank_with_zero_main_rows_still_enters_synchronized_probe_update():
    actor = make_actor(use_remove_padding=False)
    all_reduce_calls = 0

    def emulate_remote_main_rows(tensor: torch.Tensor, op):
        nonlocal all_reduce_calls
        del op
        all_reduce_calls += 1
        if tensor.ndim == 0:
            tensor.fill_(2.0)
        else:
            tensor.add_(0.25)

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.all_reduce", side_effect=emulate_remote_main_rows),
    ):
        gradient_norm = actor._update_step_value_probe(
            hidden=torch.empty(0, 4),
            targets=torch.empty(0),
            endpoint_weights=torch.empty(0),
            local_trajectory_count=0,
            global_step=1,
        )

    reductions_per_epoch = 1 + len(list(actor.step_value_probe.parameters()))
    assert all_reduce_calls == 2 * reductions_per_epoch
    assert gradient_norm > 0
    assert actor.step_value_probe_updates == 1
    assert actor.step_value_probe_warmup_completed_at == 1
    assert actor.step_value_probe_last_global_step == 1


def test_two_rank_cpu_audit_keeps_zero_main_rank_synchronized(tmp_path):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("This test requires the torch.distributed Gloo backend")
    init_method = f"file://{tmp_path / 'delayed_audit_gloo_init'}"

    torch.multiprocessing.spawn(
        run_two_rank_delayed_audit_worker,
        args=(2, init_method),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    ("missing_key", "error_pattern"),
    [
        ("step_value_probe_update_mask", "requires all actor batch keys"),
        ("step_value_prompt_center_audit_mask", "requires all actor batch keys"),
        ("step_value_forward_row_id", "requires all actor batch keys"),
    ],
)
def test_delayed_audit_rejects_partial_contract(missing_key: str, error_pattern: str):
    actor = make_actor(use_remove_padding=False)
    data = add_delayed_audit_contract(
        make_data(global_step=1),
        update_mask=torch.tensor([True, False]),
        audit_mask=torch.tensor([False, True]),
        row_ids=torch.tensor([8, 9]),
    )
    del data.batch[missing_key]

    with pytest.raises(ValueError, match=error_pattern):
        compute_step_values(actor, data)


def test_delayed_audit_rejects_nonunique_forward_row_ids():
    actor = make_actor(use_remove_padding=False)
    data = add_delayed_audit_contract(
        make_data(global_step=1),
        update_mask=torch.tensor([True, False]),
        audit_mask=torch.tensor([False, True]),
        row_ids=torch.tensor([8, 8]),
    )

    with pytest.raises(ValueError, match="must be unique"):
        compute_step_values(actor, data)


def test_repeated_global_step_reports_no_probe_update():
    actor = make_actor(use_remove_padding=False)
    compute_step_values(actor, make_data(global_step=1))
    parameters_before = [parameter.detach().clone() for parameter in actor.step_value_probe.parameters()]

    outputs = compute_step_values(actor, make_data(global_step=1))

    torch.testing.assert_close(outputs["step_value_probe_grad_norm"], torch.zeros(2))
    assert actor.step_value_probe_updates == 1
    for expected, actual in zip(parameters_before, actor.step_value_probe.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_probe_requires_global_step_metadata():
    actor = make_actor(use_remove_padding=False)
    data = make_data(global_step=1)
    del data.meta_info["global_steps"]

    with pytest.raises(ValueError, match="requires global_steps"):
        compute_step_values(actor, data)


@pytest.mark.parametrize("use_remove_padding", [False, True])
def test_step_hidden_capture_does_not_require_probe(use_remove_padding):
    actor = make_actor(use_remove_padding=use_remove_padding, probe_enabled=False)
    data = make_data(global_step=1)
    model_inputs = {**data.batch, "pad_token_id": 0}

    def cpu_logprobs(logits, labels, **_):
        return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    with (
        torch.no_grad(),
        patch("verl.workers.actor.dp_actor.logprobs_from_logits", side_effect=cpu_logprobs),
    ):
        outputs = actor._forward_micro_batch(
            model_inputs,
            temperature=1.0,
            return_step_hidden=True,
        )

    assert not actor.has_step_value_probe
    assert "step_value_hidden" not in outputs
    expected = torch.tensor([[0.3] * 4, [0.5] * 4, [0.8] * 4])
    torch.testing.assert_close(outputs["step_hidden"], expected)


@pytest.mark.parametrize("use_remove_padding", [False, True])
def test_similarity_embeddings_use_shared_step_spans_without_enabling_probe(use_remove_padding):
    actor = make_actor(use_remove_padding=use_remove_padding, probe_enabled=False)
    data = make_data(global_step=1)
    data.batch["step_start_mask"] = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )
    data.meta_info["similarity_max_steps"] = 2

    outputs = compute_similarity_embeddings(actor, data)

    assert set(outputs) == {"log_probs", "similarity_step_embeddings"}
    expected = torch.zeros(2, 2, 4, dtype=torch.float16)
    expected[0, 1] = 0.5
    expected[1, 0] = 0.5
    torch.testing.assert_close(outputs["similarity_step_embeddings"], expected)
    assert not actor.has_step_value_probe


def test_dynamic_similarity_embeddings_restore_original_trajectory_order():
    actor = make_actor(use_remove_padding=True, use_dynamic_bsz=True, probe_enabled=False)
    data = make_dynamic_data(global_step=1)
    response_mask = data.batch["attention_mask"][:, -data.batch["responses"].shape[1] :]
    data.batch["step_start_mask"] = build_step_start_mask(data.batch["step_end_mask"], response_mask)
    data.meta_info["similarity_max_steps"] = 2

    outputs = compute_similarity_embeddings(actor, data)

    expected = torch.zeros(4, 2, 4, dtype=torch.float16)
    expected[0, 1] = 0.5
    expected[2, 0] = 0.5
    torch.testing.assert_close(outputs["similarity_step_embeddings"], expected)
