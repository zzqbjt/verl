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
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _compute_step_value_diagnostics,
    compute_advantage,
)


class _PieceTokenizer:
    """Minimal reversible tokenizer with one configured source piece per ID."""

    eos_token_id = 98

    def __init__(self):
        self.pieces = {
            0: "Reason.\n\n",
            1: "Step ",
            2: "2",
            3: ": answer",
            4: "Only ",
            5: "one ",
            6: "reasoning ",
            7: "step.",
        }
        self._piece_ids = sorted(self.pieces, key=lambda token_id: len(self.pieces[token_id]), reverse=True)

    def decode(self, input_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(self.pieces[token_id] for token_id in input_ids)

    def __call__(self, text, *, add_special_tokens, return_attention_mask, return_offsets_mapping):
        assert not add_special_tokens
        assert not return_attention_mask
        assert return_offsets_mapping
        input_ids = []
        offsets = []
        char_start = 0
        while char_start < len(text):
            for token_id in self._piece_ids:
                piece = self.pieces[token_id]
                if text.startswith(piece, char_start):
                    input_ids.append(token_id)
                    offsets.append((char_start, char_start + len(piece)))
                    char_start += len(piece)
                    break
            else:
                raise AssertionError(f"No configured token piece starts at character {char_start}: {text!r}")
        return {"input_ids": input_ids, "offset_mapping": offsets}


def _step_value_config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "fsdp",
                    "ulysses_sequence_parallel_size": 1,
                    "use_prefix_grouper": False,
                    "use_fused_kernels": False,
                    "step_value_probe": {"enabled": True},
                },
                "rollout": {"multi_turn": {"enable": False}},
            },
            "algorithm": {
                "use_kl_in_reward": False,
                "rollout_correction": None,
                "step_value": {
                    "provider": "probe",
                    "target_key": "acc",
                    "task_reward_key": "score",
                    "lookahead_tokens": 3,
                    "separate_preamble": False,
                    "norm_by_group_std": True,
                },
            },
        }
    )


def _trainer(*, global_steps=17, probe_enabled=True):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = _step_value_config()
    trainer.config.actor_rollout_ref.actor.step_value_probe.enabled = probe_enabled
    trainer.use_step_value = True
    trainer.use_step_value_probe = probe_enabled
    trainer.use_legacy_worker_impl = "auto"
    trainer.tokenizer = _PieceTokenizer()
    trainer.global_steps = global_steps
    return trainer


def test_prepare_step_value_inputs_uses_non_tensor_acc_and_builds_boundaries():
    trainer = _trainer(global_steps=17)
    responses = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    response_mask = torch.ones_like(responses)
    batch = DataProto.from_dict(
        tensors={"responses": responses, "response_mask": response_mask},
        non_tensors={
            "acc": np.array([1.0, 0.0], dtype=np.float32),
            "score": np.array([1.0, -1.0], dtype=np.float32),
            "uid": np.array(["prompt", "prompt"], dtype=object),
        },
    )
    trainer._prepare_step_value_inputs(batch)

    torch.testing.assert_close(batch.batch["step_value_targets"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(batch.batch["step_value_task_rewards"], torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(batch.batch["step_value_initial_values"], torch.tensor([0.0, 1.0]))
    expected_scale = torch.tensor([1.0, 0.0]).std(unbiased=True) + 1e-6
    torch.testing.assert_close(batch.batch["step_value_scales"], expected_scale.expand(2))
    assert batch.batch["step_value_active"].tolist() == [True, True]
    expected_step_end_mask = torch.tensor(
        [
            [True, False, False, True],
            [False, False, False, True],
        ]
    )
    torch.testing.assert_close(batch.batch["step_end_mask"], expected_step_end_mask)
    assert batch.meta_info["global_steps"] == 17
    assert batch.meta_info["compute_step_value_probe"] is True


def test_prepare_step_value_inputs_allows_non_probe_provider():
    trainer = _trainer(probe_enabled=False)
    responses = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    batch = DataProto.from_dict(
        tensors={"responses": responses, "response_mask": torch.ones_like(responses)},
        non_tensors={
            "acc": np.array([1.0, 0.0], dtype=np.float32),
            "score": np.array([1.0, -1.0], dtype=np.float32),
            "uid": np.array(["prompt", "prompt"], dtype=object),
        },
    )

    trainer._prepare_step_value_inputs(batch)

    torch.testing.assert_close(batch.batch["step_value_targets"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(batch.batch["step_value_task_rewards"], torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(batch.batch["step_value_initial_values"], torch.tensor([0.0, 1.0]))
    expected_scale = torch.tensor([1.0, 0.0]).std(unbiased=True) + 1e-6
    torch.testing.assert_close(batch.batch["step_value_scales"], expected_scale.expand(2))
    assert batch.batch["step_value_active"].tolist() == [True, True]
    assert "compute_step_value_probe" not in batch.meta_info
    assert "global_steps" not in batch.meta_info


def test_similarity_provider_prepares_raw_reward_context_and_completes_estimation():
    trainer = _trainer(probe_enabled=False)
    trainer.config.algorithm.step_value.provider = "similarity"
    trainer.step_value_provider = "similarity"
    responses = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    response_mask = torch.ones_like(responses)
    token_rewards = torch.zeros_like(responses, dtype=torch.float32)
    token_rewards[:, -1] = torch.tensor([-2.0, 3.0])
    batch = DataProto.from_dict(
        tensors={
            "responses": responses,
            "response_mask": response_mask,
            "token_level_rewards": token_rewards,
        },
        non_tensors={"uid": np.array(["prompt", "prompt"], dtype=object)},
    )

    trainer._prepare_step_value_inputs(batch)

    torch.testing.assert_close(batch.batch["step_value_targets"], torch.tensor([-2.0, 3.0]))
    torch.testing.assert_close(batch.batch["step_value_task_rewards"], torch.tensor([-2.0, 3.0]))
    torch.testing.assert_close(batch.batch["step_value_initial_values"], torch.tensor([3.0, -2.0]))
    expected_scale = torch.tensor([-2.0, 3.0]).std(unbiased=True) + 1e-6
    torch.testing.assert_close(batch.batch["step_value_scales"], expected_scale.expand(2))
    assert "compute_step_value_probe" not in batch.meta_info
    assert batch.meta_info["compute_similarity_step_embeddings"] is True
    assert batch.meta_info["similarity_max_steps"] == 2

    batch.batch["similarity_step_embeddings"] = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    trainer._complete_step_value_estimation(batch)

    expected_values = torch.zeros_like(responses, dtype=torch.float32)
    expected_values[0, 0] = 3.0
    expected_values[0, 3] = -2.0
    expected_values[1, 3] = 3.0
    torch.testing.assert_close(batch.batch["step_values"], expected_values)
    assert batch.batch["step_value_ready"].tolist() == [True, True]
    assert "similarity_step_embeddings" not in batch.batch
    assert "compute_similarity_step_embeddings" not in batch.meta_info

    compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.STEP_VALUE,
        config=trainer.config.algorithm,
    )
    assert batch.batch["advantages"].shape == responses.shape
    assert torch.isfinite(batch.batch["advantages"]).all()


def test_prepare_step_inputs_does_not_require_step_value_or_probe():
    trainer = _trainer()
    trainer.use_step_value = False
    trainer.use_step_split = True
    trainer.config.actor_rollout_ref.actor.step_value_probe.enabled = False
    trainer.config.algorithm.step_split = OmegaConf.create(
        {"enabled": True, "lookahead_tokens": 3, "separate_preamble": False}
    )
    responses = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    batch = DataProto.from_dict(tensors={"responses": responses, "response_mask": torch.ones_like(responses)})

    trainer._prepare_step_inputs(batch)

    expected_step_end_mask = torch.tensor(
        [
            [True, False, False, True],
            [False, False, False, True],
        ]
    )
    torch.testing.assert_close(batch.batch["step_end_mask"], expected_step_end_mask)
    assert "step_value_targets" not in batch.batch
    assert "compute_step_value_probe" not in batch.meta_info


def test_prepare_step_inputs_is_automatic_for_vinfo():
    trainer = _trainer()
    trainer.use_step_value = False
    if hasattr(trainer, "use_step_split"):
        del trainer.use_step_split
    trainer.config.algorithm.adv_estimator = AdvantageEstimator.V_INFO.value
    trainer.config.actor_rollout_ref.actor.step_value_probe.enabled = False
    trainer.config.algorithm.step_split = OmegaConf.create(
        {"enabled": False, "lookahead_tokens": 3, "separate_preamble": False}
    )
    responses = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    batch = DataProto.from_dict(tensors={"responses": responses, "response_mask": torch.ones_like(responses)})

    trainer._prepare_step_inputs(batch)

    expected_step_end_mask = torch.tensor(
        [
            [True, False, False, True],
            [False, False, False, True],
        ]
    )
    torch.testing.assert_close(batch.batch["step_end_mask"], expected_step_end_mask)


def test_finalize_step_value_batch_records_metrics_and_removes_temporary_tensors():
    trainer = _trainer()
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones(2, 4, dtype=torch.long),
            "advantages": torch.ones(2, 4),
            "step_values": torch.zeros(2, 4),
            "step_end_mask": torch.tensor(
                [
                    [True, False, False, True],
                    [False, False, False, True],
                ]
            ),
            "step_value_targets": torch.tensor([1.0, 0.0]),
            "step_value_task_rewards": torch.tensor([1.0, -1.0]),
            "step_value_initial_values": torch.tensor([0.0, 1.0]),
            "step_value_scales": torch.tensor([0.5, 0.5]),
            "step_value_active": torch.tensor([True, True]),
            "step_value_ready": torch.tensor([False, True]),
            "step_value_trajectory_logit_mean": torch.tensor([-0.2, 0.3]),
            "step_value_probe_loss": torch.tensor([0.2, 0.6]),
            "step_value_probe_grad_norm": torch.tensor([0.4, 0.4]),
        }
    )
    batch.meta_info.update({"compute_step_value_probe": True, "global_steps": 17})
    metrics = {"actor/loss": 1.0}

    trainer._finalize_step_value_batch(batch, metrics)

    assert metrics["actor/loss"] == 1.0
    assert metrics["step_value/probe_loss"] == pytest.approx(0.4)
    assert metrics["step_value/ready"] == pytest.approx(0.5)
    assert metrics["step_value/probe_grad_norm"] == pytest.approx(0.4)
    assert metrics["step_value/avg_steps_per_response"] == pytest.approx(1.5)
    assert metrics["step_value/preupdate_brier_skill"] == pytest.approx(0.0)
    assert metrics["step_value/preupdate_nonterminal_coverage"] == pytest.approx(0.5)
    assert set(metrics) == {
        "actor/loss",
        "step_value/probe_loss",
        "step_value/ready",
        "step_value/probe_grad_norm",
        "step_value/avg_steps_per_response",
        "step_value/target_positive_fraction",
        "step_value/preupdate_nonterminal_coverage",
        "step_value/preupdate_accuracy",
        "step_value/preupdate_brier_skill",
        "step_value/preupdate_ece_10",
        "step_value/value_separation",
        "step_value/depth_q1_brier",
        "step_value/depth_q4_brier",
        "step_value/delta_abs_mean",
        "step_value/delta_near_zero_fraction",
    }
    for key in (
        "step_values",
        "step_end_mask",
        "step_value_targets",
        "step_value_task_rewards",
        "step_value_initial_values",
        "step_value_scales",
        "step_value_active",
        "step_value_ready",
        "step_value_trajectory_logit_mean",
        "step_value_probe_loss",
        "step_value_probe_grad_norm",
    ):
        assert key not in batch.batch
    assert "responses" in batch.batch
    assert "advantages" in batch.batch
    assert "compute_step_value_probe" not in batch.meta_info
    assert "global_steps" not in batch.meta_info


def test_finalize_step_value_batch_diagnostics_exclude_terminal_anchor():
    trainer = _trainer()
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones(2, 2, dtype=torch.long),
            "advantages": torch.ones(2, 2),
            # Nonterminal predictions are wrong, while terminal values are the
            # exact Y anchors and must not inflate the diagnostic accuracy.
            "step_values": torch.tensor([[0.1, 1.0], [0.9, 0.0]]),
            "step_end_mask": torch.ones(2, 2, dtype=torch.bool),
            "step_value_targets": torch.tensor([1.0, 0.0]),
        }
    )

    metrics = {}
    trainer._finalize_step_value_batch(batch, metrics)

    assert metrics["step_value/preupdate_accuracy"] == pytest.approx(0.0)
    assert metrics["step_value/preupdate_nonterminal_coverage"] == pytest.approx(1.0)


def test_step_value_diagnostics_measure_preupdate_calibration_separation_and_deltas():
    diagnostics = _compute_step_value_diagnostics(
        step_values=torch.tensor(
            [
                [0.2, 0.0, 0.0, 0.8],
                [0.1, 0.0, 0.0, 0.3],
            ]
        ),
        step_end_mask=torch.tensor(
            [
                [True, False, False, True],
                [True, False, False, True],
            ]
        ),
        targets=torch.tensor([1.0, 0.0]),
    )

    assert set(diagnostics) == {
        "step_value/avg_steps_per_response",
        "step_value/target_positive_fraction",
        "step_value/preupdate_nonterminal_coverage",
        "step_value/preupdate_accuracy",
        "step_value/preupdate_brier_skill",
        "step_value/preupdate_ece_10",
        "step_value/value_separation",
        "step_value/depth_q1_brier",
        "step_value/depth_q4_brier",
        "step_value/delta_abs_mean",
        "step_value/delta_near_zero_fraction",
    }
    assert diagnostics["step_value/avg_steps_per_response"] == pytest.approx(2.0)
    assert diagnostics["step_value/target_positive_fraction"] == pytest.approx(0.5)
    assert diagnostics["step_value/preupdate_nonterminal_coverage"] == pytest.approx(1.0)
    assert diagnostics["step_value/preupdate_accuracy"] == pytest.approx(0.5)
    assert diagnostics["step_value/preupdate_brier_skill"] == pytest.approx(-0.3)
    assert diagnostics["step_value/preupdate_ece_10"] == pytest.approx(0.45)
    assert diagnostics["step_value/value_separation"] == pytest.approx(0.1)
    assert diagnostics["step_value/delta_abs_mean"] == pytest.approx(0.0)
    assert diagnostics["step_value/delta_near_zero_fraction"] == pytest.approx(0.0)
    assert diagnostics["step_value/depth_q1_brier"] == pytest.approx(0.325)
    assert diagnostics["step_value/depth_q4_brier"] == pytest.approx(0.325)


def test_step_value_diagnostics_stay_finite_for_one_class_and_one_step():
    diagnostics = _compute_step_value_diagnostics(
        step_values=torch.tensor([[0.9, 0.0], [0.0, 0.7]]),
        step_end_mask=torch.tensor([[True, False], [False, True]]),
        targets=torch.ones(2),
    )

    assert diagnostics["step_value/preupdate_brier_skill"] == pytest.approx(0.0)
    assert diagnostics["step_value/value_separation"] == pytest.approx(0.0)
    assert diagnostics["step_value/delta_abs_mean"] == pytest.approx(0.0)
    assert diagnostics["step_value/avg_steps_per_response"] == pytest.approx(1.0)
    assert diagnostics["step_value/preupdate_nonterminal_coverage"] == pytest.approx(0.0)
    assert all(np.isfinite(value) for value in diagnostics.values())


def test_step_value_diagnostics_weight_each_trajectory_equally():
    diagnostics = _compute_step_value_diagnostics(
        step_values=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        step_end_mask=torch.tensor(
            [
                [True, False, True],
                [True, True, True],
            ]
        ),
        targets=torch.tensor([1.0, 0.0]),
    )

    # The first trajectory has Brier error 1 and the second has error 0,
    # giving skill -1. Endpoint weighting would incorrectly give skill 0.
    assert diagnostics["step_value/preupdate_brier_skill"] == pytest.approx(-1.0)
    assert diagnostics["step_value/avg_steps_per_response"] == pytest.approx(2.5)
    assert diagnostics["step_value/preupdate_nonterminal_coverage"] == pytest.approx(1.0)


def test_step_value_diagnostics_reject_soft_targets():
    with pytest.raises(ValueError, match="binary values"):
        _compute_step_value_diagnostics(
            step_values=torch.tensor([[0.5]]),
            step_end_mask=torch.ones(1, 1, dtype=torch.bool),
            targets=torch.tensor([0.5]),
        )


def test_step_value_runtime_validation_accepts_supported_path():
    trainer = _trainer()

    trainer._validate_step_value_runtime()
    trainer._validate_step_value_probe_runtime()


def test_step_value_runtime_does_not_require_probe():
    trainer = _trainer(probe_enabled=False)

    trainer._validate_step_value_runtime()


def test_similarity_provider_is_validated_by_the_standard_trainer():
    trainer = _trainer(probe_enabled=False)
    trainer.step_value_provider = "similarity"
    trainer.config.algorithm.step_value.provider = "similarity"
    trainer.config.actor_rollout_ref.rollout.n = 8

    trainer._validate_step_value_runtime()

    trainer.config.actor_rollout_ref.actor.step_value_probe.enabled = True
    with pytest.raises(ValueError, match="step_value_probe.enabled=false"):
        trainer._validate_step_value_runtime()


@pytest.mark.parametrize(
    ("config_path", "value", "legacy_mode", "message"),
    [
        (None, None, "disable", "requires the legacy FSDP actor worker API"),
        ("actor_rollout_ref.actor.strategy", "megatron", "auto", "supports only actor strategy"),
        (
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size",
            2,
            "auto",
            "requires ulysses_sequence_parallel_size=1",
        ),
        ("actor_rollout_ref.actor.use_prefix_grouper", True, "auto", "not compatible with use_prefix_grouper"),
        ("actor_rollout_ref.actor.use_fused_kernels", True, "auto", "not yet validated with use_fused_kernels"),
        ("algorithm.use_kl_in_reward", True, "auto", "requires algorithm.use_kl_in_reward=false"),
        ("algorithm.rollout_correction", {"bypass_mode": True}, "auto", "cannot use bypass_mode"),
        ("actor_rollout_ref.rollout.multi_turn.enable", True, "auto", "supports single-turn text rollouts only"),
    ],
)
def test_step_value_runtime_validation_rejects_unsupported_paths(config_path, value, legacy_mode, message):
    trainer = _trainer()
    trainer.use_legacy_worker_impl = legacy_mode
    if config_path is not None:
        OmegaConf.update(trainer.config, config_path, value, merge=False)

    with pytest.raises(ValueError, match=message):
        trainer._validate_step_value_probe_runtime()


def test_compute_advantage_passes_fixed_lambda_and_step_value_std_option(monkeypatch):
    trainer = _trainer(global_steps=2)
    trainer.config.algorithm.step_value.lam = 0.83
    shape = (2, 3)
    batch = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.zeros(shape),
            "response_mask": torch.ones(shape),
            "step_values": torch.zeros(shape),
            "step_end_mask": torch.ones(shape, dtype=torch.bool),
            "step_value_initial_values": torch.tensor([0.0, 1.0]),
            "step_value_scales": torch.tensor([0.5, 0.5]),
            "step_value_active": torch.tensor([True, True]),
            "step_value_task_rewards": torch.tensor([1.0, -1.0]),
            "step_value_ready": torch.ones(2, dtype=torch.bool),
        },
        non_tensors={"uid": np.array(["a", "b"], dtype=object)},
    )
    captured = {}

    def fake_step_value_advantage(**kwargs):
        captured.update(kwargs)
        return torch.full(shape, 2.0), torch.full(shape, 3.0)

    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.core_algos.compute_step_value_advantage",
        fake_step_value_advantage,
    )

    compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.STEP_VALUE,
        config=trainer.config.algorithm,
    )

    assert captured["lam"] == pytest.approx(0.83)
    assert captured["norm_by_group_std"] is True
    torch.testing.assert_close(captured["step_value_initial_values"], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(captured["step_value_scales"], torch.tensor([0.5, 0.5]))
    assert captured["step_value_active"].tolist() == [True, True]
    assert "norm_adv_by_std_in_grpo" not in captured
    assert "step_value_overlong_rewards" not in captured
    torch.testing.assert_close(batch.batch["advantages"], torch.full(shape, 2.0))
    torch.testing.assert_close(batch.batch["returns"], torch.full(shape, 3.0))
