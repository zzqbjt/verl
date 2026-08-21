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

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    compute_step_value_advantage,
    prepare_step_value_context,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from verl.trainer.ppo.step_value_prompt_center import (
    LaggedPromptCenterAuditState,
    aggregate_prompt_center_audit_groups,
    apply_rank_preserving_prompt_center_calibration,
    fit_group_binomial_prompt_center_map,
    prompt_center_logit_offsets,
)

SLOPE = 3.0839867548938047
INTERCEPT = -0.2211958702620491


def _calibration_example(dtype: torch.dtype = torch.float64):
    prompt_ids = np.array(["p", "q", "p", "q"], dtype=object)
    step_end_mask = torch.tensor(
        [
            [True, False, True, False, True],
            [False, True, False, False, True],
            [True, False, False, True, True],
            [False, False, True, False, True],
        ]
    )
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=dtype)
    predicted_logits = (
        (-2.0, 0.5),
        (-0.3,),
        (1.2, -1.0),
        (0.7,),
    )
    raw_terminal_logits = (0.1, -0.4, 0.8, 0.2)
    step_values = torch.zeros(step_end_mask.shape, dtype=dtype)
    trajectory_means = torch.empty(4, dtype=dtype)
    for row in range(4):
        positions = torch.nonzero(step_end_mask[row], as_tuple=False).flatten()
        nonterminal_positions = positions[:-1]
        row_logits = torch.tensor(predicted_logits[row], dtype=dtype)
        step_values[row, nonterminal_positions] = torch.sigmoid(row_logits)
        step_values[row, positions[-1]] = targets[row]
        trajectory_means[row] = torch.cat((row_logits, torch.tensor([raw_terminal_logits[row]], dtype=dtype))).mean()
    return step_values, step_end_mask, trajectory_means, prompt_ids, targets


def _apply(example, *, enabled: bool = True):
    values, end_mask, means, prompt_ids, targets = example
    return apply_rank_preserving_prompt_center_calibration(
        values,
        end_mask,
        means,
        prompt_ids,
        targets,
        enabled=enabled,
        slope=SLOPE,
        intercept=INTERCEPT,
    )


def test_shared_prompt_offset_preserves_all_predicted_logit_differences() -> None:
    example = _calibration_example()
    values, end_mask, _, prompt_ids, _ = example
    calibrated, metrics = _apply(example)

    for prompt_id in ("p", "q"):
        rows = np.flatnonzero(prompt_ids == prompt_id)
        predicted_mask = end_mask[rows].clone()
        for local_row in range(len(rows)):
            positions = torch.nonzero(predicted_mask[local_row], as_tuple=False).flatten()
            predicted_mask[local_row, positions[-1]] = False
        before = torch.logit(values[rows][predicted_mask])
        after = torch.logit(calibrated[rows][predicted_mask])
        torch.testing.assert_close(
            before[:, None] - before[None, :],
            after[:, None] - after[None, :],
            atol=1e-12,
            rtol=1e-12,
        )

    assert metrics["enabled"] is True
    assert metrics["prompt_count"] == 2
    assert metrics["predicted_endpoint_count"] == 6


def test_disabled_calibration_is_an_exact_identity() -> None:
    example = _calibration_example(dtype=torch.float32)
    values = example[0]
    calibrated, metrics = _apply(example, enabled=False)

    assert calibrated is values
    assert torch.equal(calibrated, values)
    assert metrics == {"enabled": False}


def test_enabled_identity_map_preserves_nonterminal_values_bitwise() -> None:
    values, end_mask, means, prompt_ids, targets = _calibration_example(dtype=torch.float32)

    calibrated, metrics = apply_rank_preserving_prompt_center_calibration(
        values,
        end_mask,
        means,
        prompt_ids,
        targets,
        enabled=True,
        slope=1.0,
        intercept=0.0,
    )

    assert torch.equal(calibrated, values)
    assert metrics["enabled"] is True
    assert metrics["offset_min"] == 0.0
    assert metrics["offset_max"] == 0.0


def test_prompt_group_reordering_keeps_response_alignment() -> None:
    example = _calibration_example()
    expected, _ = _apply(example)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    values, end_mask, means, prompt_ids, targets = example
    reordered = (
        values[permutation],
        end_mask[permutation],
        means[permutation],
        prompt_ids[permutation.numpy()],
        targets[permutation],
    )

    actual_reordered, _ = _apply(reordered)

    torch.testing.assert_close(actual_reordered[inverse], expected, atol=1e-12, rtol=1e-12)


def test_prompt_center_is_trajectory_balanced_and_terminal_targets_are_restored() -> None:
    values, end_mask, means, prompt_ids, targets = _calibration_example()
    offsets, centers = prompt_center_logit_offsets(
        means,
        prompt_ids,
        slope=SLOPE,
        intercept=INTERCEPT,
    )
    expected_p_center = means[torch.tensor([0, 2])].mean()
    expected_q_center = means[torch.tensor([1, 3])].mean()
    torch.testing.assert_close(centers.sort().values, torch.stack((expected_p_center, expected_q_center)).sort().values)
    torch.testing.assert_close(offsets[0], offsets[2])
    torch.testing.assert_close(offsets[1], offsets[3])

    corrupted = values.clone()
    terminal_positions = torch.arange(values.shape[1]).expand_as(end_mask).masked_fill(~end_mask, -1).max(dim=-1).values
    corrupted[torch.arange(4), terminal_positions] = 0.37
    calibrated, _ = _apply((corrupted, end_mask, means, prompt_ids, targets))

    torch.testing.assert_close(calibrated[torch.arange(4), terminal_positions], targets)
    assert torch.equal(calibrated[~end_mask], torch.zeros_like(calibrated[~end_mask]))


def test_driver_calibration_flows_into_step_value_advantages() -> None:
    values, end_mask, means, prompt_ids, targets = _calibration_example(dtype=torch.float32)
    response_mask = torch.ones_like(values)
    task_rewards = targets.clone()
    token_rewards = torch.zeros_like(values)
    token_rewards[:, -1] = task_rewards
    initial_values, scales, active = prepare_step_value_context(
        targets,
        prompt_ids,
        norm_by_group_std=True,
        zero_when_group_uniform=False,
    )
    batch = DataProto.from_dict(
        tensors={
            "token_level_rewards": token_rewards,
            "response_mask": response_mask,
            "step_values": values.clone(),
            "step_end_mask": end_mask,
            "step_value_trajectory_logit_mean": means,
            "step_value_targets": targets,
            "step_value_initial_values": initial_values,
            "step_value_scales": scales,
            "step_value_active": active,
            "step_value_task_rewards": task_rewards,
            "step_value_ready": torch.ones(4, dtype=torch.bool),
        },
        non_tensors={"uid": prompt_ids},
    )
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.use_step_value = True
    trainer.step_value_provider = "probe"
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "step_value": {
                    "provider": "probe",
                    "lam": 0.9,
                    "norm_by_group_std": True,
                    "prompt_center_calibration_enabled": True,
                    "prompt_center_calibration_slope": SLOPE,
                    "prompt_center_calibration_intercept": INTERCEPT,
                }
            }
        }
    )
    expected_values, _ = _apply((values, end_mask, means, prompt_ids, targets))

    trainer._complete_step_value_estimation(batch)
    torch.testing.assert_close(batch.batch["step_values"], expected_values)
    compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.STEP_VALUE,
        config=trainer.config.algorithm,
    )

    expected_advantages, _ = compute_step_value_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=prompt_ids,
        step_values=expected_values,
        step_end_mask=end_mask,
        step_value_initial_values=initial_values,
        step_value_scales=scales,
        step_value_active=active,
        step_value_task_rewards=task_rewards,
        step_value_ready=True,
        lam=0.9,
        norm_by_group_std=True,
    )
    torch.testing.assert_close(batch.batch["advantages"], expected_advantages)

    baseline_advantages, _ = compute_step_value_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=prompt_ids,
        step_values=values,
        step_end_mask=end_mask,
        step_value_initial_values=initial_values,
        step_value_scales=scales,
        step_value_active=active,
        step_value_task_rewards=task_rewards,
        step_value_ready=True,
        lam=0.9,
        norm_by_group_std=True,
    )
    assert not torch.allclose(batch.batch["advantages"], baseline_advantages)
    assert batch.meta_info["prompt_center_calibration_metrics"]["enabled"] is True


def test_driver_disabled_path_does_not_require_calibration_inputs() -> None:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.use_step_value = True
    trainer.step_value_provider = "probe"
    trainer.config = OmegaConf.create(
        {"algorithm": {"step_value": {"provider": "probe", "prompt_center_calibration_enabled": False}}}
    )
    batch = DataProto.from_dict(tensors={"step_values": torch.tensor([[0.25]])})
    original = batch.batch["step_values"].clone()

    trainer._complete_step_value_estimation(batch)

    torch.testing.assert_close(batch.batch["step_values"], original, atol=0.0, rtol=0.0)
    assert "prompt_center_calibration_metrics" not in batch.meta_info


def test_invalid_nonpositive_slope_is_rejected() -> None:
    example = _calibration_example()
    with pytest.raises(ValueError, match="slope"):
        apply_rank_preserving_prompt_center_calibration(
            *example,
            enabled=True,
            slope=0.0,
            intercept=INTERCEPT,
        )


def test_group_binomial_fit_recovers_a_known_affine_map() -> None:
    centers = np.linspace(-2.0, 2.0, 41)
    expected_slope = 1.8
    expected_intercept = -0.35
    probabilities = 1.0 / (1.0 + np.exp(-(expected_slope * centers + expected_intercept)))
    totals = np.full(centers.shape, 200, dtype=np.int64)
    successes = np.rint(totals * probabilities).astype(np.int64)

    fit = fit_group_binomial_prompt_center_map(
        centers,
        successes,
        totals,
        previous_slope=1.0,
        previous_intercept=0.0,
    )

    assert fit.updated is True
    assert fit.reason == "fit"
    assert fit.slope == pytest.approx(expected_slope, abs=0.01)
    assert fit.intercept == pytest.approx(expected_intercept, abs=0.01)


def test_separated_or_degenerate_audit_retains_previous_parameters() -> None:
    separated = fit_group_binomial_prompt_center_map(
        [-2.0, -1.0, 1.0, 2.0],
        [0, 0, 8, 8],
        [8, 8, 8, 8],
        previous_slope=1.25,
        previous_intercept=-0.2,
    )
    constant = fit_group_binomial_prompt_center_map(
        [0.4, 0.4, 0.4],
        [1, 4, 7],
        [8, 8, 8],
        previous_slope=1.25,
        previous_intercept=-0.2,
    )

    assert separated.reason == "positive_slope_separation"
    assert constant.reason == "constant_center"
    assert separated.updated is constant.updated is False
    assert separated.slope == constant.slope == 1.25
    assert separated.intercept == constant.intercept == -0.2


def test_audit_group_aggregation_is_trajectory_balanced_and_complete() -> None:
    prompt_ids = np.asarray(["b", "a", "b", "a"], dtype=object)
    means = torch.tensor([3.0, -2.0, 1.0, 4.0])
    endpoint_counts = torch.tensor([10, 1, 2, 8])
    ready = torch.ones(4, dtype=torch.bool)
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0])

    centers, successes, totals = aggregate_prompt_center_audit_groups(
        means,
        endpoint_counts,
        ready,
        prompt_ids,
        targets,
        expected_group_size=2,
    )

    # Groups are sorted by UID; endpoint counts deliberately differ and do not
    # weight the trajectory means.
    np.testing.assert_allclose(centers, np.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(successes, np.asarray([1, 1]))
    np.testing.assert_array_equal(totals, np.asarray([2, 2]))


def test_lagged_audit_state_requires_explicit_commit_and_round_trips() -> None:
    state = LaggedPromptCenterAuditState(
        initial_slope=1.0,
        initial_intercept=0.0,
        target_key="acc",
        group_size=8,
        audit_groups=16,
        rolling_window=2,
        seed=17,
    )
    centers = np.linspace(-1.5, 1.5, 16)
    totals = np.full(16, 8, dtype=np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-(1.7 * centers - 0.3)))
    successes = np.clip(np.rint(totals * probabilities), 0, totals).astype(np.int64)

    fit = state.stage(centers=centers, successes=successes, totals=totals, global_step=1)
    assert fit.updated is True
    assert state.active_parameters == (1.0, 0.0)
    with pytest.raises(RuntimeError, match="update boundary"):
        state.state_dict()

    next_parameters = state.commit()
    assert next_parameters != (1.0, 0.0)
    state.stage(centers=centers + 0.1, successes=successes, totals=totals, global_step=2)
    state.commit()
    serialized = state.state_dict()

    restored = LaggedPromptCenterAuditState(
        initial_slope=1.0,
        initial_intercept=0.0,
        target_key="acc",
        group_size=8,
        audit_groups=16,
        rolling_window=2,
        seed=17,
    )
    restored.load_state_dict(serialized)
    assert restored.state_dict() == serialized

    malformed_states = []
    for field, invalid_value in (
        ("format_version", "1"),
        ("active_slope", 1),
        ("last_audit_global_step", True),
        ("fit_count", 2.0),
    ):
        malformed = deepcopy(serialized)
        malformed[field] = invalid_value
        malformed_states.append(malformed)
    malformed = deepcopy(serialized)
    malformed["fingerprint"]["group_size"] = True
    malformed_states.append(malformed)
    malformed = deepcopy(serialized)
    malformed["entries"][0]["successes"][0] = 0.5
    malformed_states.append(malformed)
    malformed = deepcopy(serialized)
    malformed["entries"][0]["centers"][0] = 0
    malformed_states.append(malformed)
    for malformed in malformed_states:
        strict_restore = LaggedPromptCenterAuditState(
            initial_slope=1.0,
            initial_intercept=0.0,
            target_key="acc",
            group_size=8,
            audit_groups=16,
            rolling_window=2,
            seed=17,
        )
        with pytest.raises(ValueError):
            strict_restore.load_state_dict(malformed)

    mismatched = LaggedPromptCenterAuditState(
        initial_slope=1.0,
        initial_intercept=0.0,
        target_key="acc",
        group_size=8,
        audit_groups=16,
        rolling_window=1,
        seed=17,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        mismatched.load_state_dict(serialized)


def test_first_batch_uid_audit_selection_is_stable_under_outcome_permutation() -> None:
    from recipe.dapo.dapo_ray_trainer import select_stable_prompt_center_audit_uids

    prompt_ids = np.repeat(np.asarray([f"prompt-{index}" for index in range(32)], dtype=object), 8)
    outcomes = np.tile(np.asarray([0, 1, 0, 1, 0, 1, 0, 1]), 32)
    expected = select_stable_prompt_center_audit_uids(
        prompt_ids,
        group_size=8,
        audit_groups=16,
        seed=11,
        global_step=7,
    )
    # Reorder every trajectory using an outcome-derived permutation.  The
    # selected UID set remains identical because rewards never enter priority.
    permutation = np.lexsort((np.arange(prompt_ids.size), outcomes))
    actual = select_stable_prompt_center_audit_uids(
        prompt_ids[permutation],
        group_size=8,
        audit_groups=16,
        seed=11,
        global_step=7,
    )

    assert set(actual.tolist()) == set(expected.tolist())
    assert len(actual) == 16

    replacement_ids = np.repeat(np.asarray([f"replacement-{index}" for index in range(32)], dtype=object), 8)
    replacement = select_stable_prompt_center_audit_uids(
        replacement_ids,
        group_size=8,
        audit_groups=16,
        seed=11,
        global_step=7,
    )
    expected_ordinals = {int(uid.rsplit("-", 1)[1]) for uid in expected.tolist()}
    replacement_ordinals = {int(uid.rsplit("-", 1)[1]) for uid in replacement.tolist()}
    assert replacement_ordinals == expected_ordinals


def test_audit_checkpoint_hooks_fail_closed_and_restore_exact_state(tmp_path) -> None:
    from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.use_step_value_prompt_center_audit = True
    trainer.global_steps = 3
    trainer._prompt_center_audit_state = LaggedPromptCenterAuditState(
        initial_slope=1.0,
        initial_intercept=0.0,
        target_key="acc",
        group_size=8,
        audit_groups=16,
        rolling_window=2,
        seed=0,
    )
    trainer._prompt_center_audit_state.stage_unchanged(global_step=3, reason="probe_not_ready")
    trainer._prompt_center_audit_state.commit()
    trainer._save_trainer_extra_state(str(tmp_path))

    restored = RayDAPOTrainer.__new__(RayDAPOTrainer)
    restored.use_step_value_prompt_center_audit = True
    restored.global_steps = 3
    restored.config = OmegaConf.create({"algorithm": {"step_value": {}}})
    restored._prompt_center_audit_state = LaggedPromptCenterAuditState(
        initial_slope=1.0,
        initial_intercept=0.0,
        target_key="acc",
        group_size=8,
        audit_groups=16,
        rolling_window=2,
        seed=0,
    )
    restored._load_trainer_extra_state(str(tmp_path))
    assert restored._prompt_center_audit_state.state_dict() == trainer._prompt_center_audit_state.state_dict()

    for invalid_checkpoint_step in (True, 3.0, "3", 4):
        invalid_step_dir = tmp_path / f"invalid-checkpoint-step-{invalid_checkpoint_step!r}"
        invalid_step_dir.mkdir()
        torch.save(
            {
                "checkpoint_global_step": invalid_checkpoint_step,
                "audit_state": trainer._prompt_center_audit_state.state_dict(),
            },
            invalid_step_dir / "step_value_prompt_center_audit.pt",
        )
        invalid_step = RayDAPOTrainer.__new__(RayDAPOTrainer)
        invalid_step.use_step_value_prompt_center_audit = True
        invalid_step.global_steps = 3
        invalid_step.config = restored.config
        invalid_step._prompt_center_audit_state = LaggedPromptCenterAuditState(
            initial_slope=1.0,
            initial_intercept=0.0,
            target_key="acc",
            group_size=8,
            audit_groups=16,
            rolling_window=2,
            seed=0,
        )
        with pytest.raises(RuntimeError, match="global step is missing or mismatched"):
            invalid_step._load_trainer_extra_state(str(invalid_step_dir))

    missing = RayDAPOTrainer.__new__(RayDAPOTrainer)
    missing.use_step_value_prompt_center_audit = True
    missing.global_steps = 3
    missing.config = restored.config
    missing._prompt_center_audit_state = LaggedPromptCenterAuditState(
        initial_slope=1.0,
        initial_intercept=0.0,
        target_key="acc",
        group_size=8,
        audit_groups=16,
        rolling_window=2,
        seed=0,
    )
    with pytest.raises(RuntimeError, match="missing"):
        missing._load_trainer_extra_state(str(tmp_path / "does-not-exist"))

    invalid_path_dir = tmp_path / "invalid-path"
    invalid_path_dir.mkdir()
    (invalid_path_dir / "step_value_prompt_center_audit.pt").mkdir()
    invalid_path = RayDAPOTrainer.__new__(RayDAPOTrainer)
    invalid_path.use_step_value_prompt_center_audit = True
    invalid_path.global_steps = 83
    invalid_path.config = restored.config
    invalid_path._prompt_center_audit_state = LaggedPromptCenterAuditState(
        initial_slope=1.0,
        initial_intercept=0.0,
        target_key="acc",
        group_size=8,
        audit_groups=16,
        rolling_window=2,
        seed=0,
    )
    with pytest.raises(RuntimeError, match="not a regular file"):
        invalid_path._load_trainer_extra_state(str(invalid_path_dir))

    malformed_dir = tmp_path / "malformed"
    malformed_dir.mkdir()
    torch.save(
        {"checkpoint_global_step": 83, "audit_state": {"format_version": -1}},
        malformed_dir / "step_value_prompt_center_audit.pt",
    )
    malformed = RayDAPOTrainer.__new__(RayDAPOTrainer)
    malformed.use_step_value_prompt_center_audit = True
    malformed.global_steps = 83
    malformed.config = restored.config
    malformed._prompt_center_audit_state = LaggedPromptCenterAuditState(
        initial_slope=1.0,
        initial_intercept=0.0,
        target_key="acc",
        group_size=8,
        audit_groups=16,
        rolling_window=2,
        seed=0,
    )
    with pytest.raises(RuntimeError, match="Invalid"):
        malformed._load_trainer_extra_state(str(malformed_dir))


@pytest.mark.parametrize(
    ("filter_metric", "use_kl", "message"),
    [
        ("score", False, "filter_groups.metric"),
        ("acc", True, "use_kl_in_reward=false"),
    ],
)
def test_dapo_audit_runtime_rejects_semantically_incompatible_paths(
    filter_metric: str,
    use_kl: bool,
    message: str,
) -> None:
    from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.step_value_provider = "probe"
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "use_kl_in_reward": use_kl,
                "step_value": {
                    "provider": "probe",
                    "target_key": "acc",
                    "prompt_center_calibration_enabled": False,
                    "prompt_center_audit_enabled": True,
                    "prompt_center_audit_groups": 16,
                },
                "filter_groups": {"enable": True, "metric": filter_metric},
            },
            "actor_rollout_ref": {"rollout": {"n": 8}, "actor": {"step_value_probe": {"enabled": True}}},
            "data": {"train_batch_size": 128},
        }
    )

    with pytest.raises(ValueError, match=message):
        trainer._validate_step_value_runtime()


def test_dapo_audit_runtime_rejects_a_disabled_probe() -> None:
    from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.step_value_provider = "probe"
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "use_kl_in_reward": False,
                "step_value": {
                    "provider": "probe",
                    "target_key": "acc",
                    "prompt_center_audit_enabled": True,
                },
                "filter_groups": {"enable": True, "metric": "acc"},
            },
            "actor_rollout_ref": {"rollout": {"n": 8}, "actor": {"step_value_probe": {"enabled": False}}},
            "data": {"train_batch_size": 128},
        }
    )

    with pytest.raises(ValueError, match="step_value_probe.enabled=true"):
        trainer._validate_step_value_runtime()


def test_dapo_main_collection_excludes_audit_uids_and_refills_from_later_batches() -> None:
    from recipe.dapo.dapo_ray_trainer import select_dapo_main_prompt_uids

    audit_uids = {"audit-0", "audit-1"}
    generation_batches = [
        {
            "audit-0": [0.0, 1.0],
            "audit-1": [1.0, 0.0],
            "main-0": [0.0, 1.0],
            "uniform": [1.0, 1.0],
        },
        {
            "main-1": [0.0, 1.0],
            "main-2": [1.0, 0.0],
            "main-3": [0.0, 1.0],
        },
    ]
    collected: list[object] = []
    batches_used = 0
    for candidates in generation_batches:
        batches_used += 1
        collected.extend(select_dapo_main_prompt_uids(candidates, excluded_uids=audit_uids))
        if len(collected) >= 3:
            break

    assert batches_used == 2
    assert collected[:3] == ["main-0", "main-1", "main-2"]
    assert not (set(collected) & audit_uids)


def test_combined_dapo_forward_is_disjoint_balanced_and_causally_lagged() -> None:
    from types import SimpleNamespace

    from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

    group_size = 8
    main_group_uids = np.asarray([f"main-{index}" for index in range(16)], dtype=object)
    audit_uids = np.asarray([f"audit-{index}" for index in range(16)], dtype=object)

    def grouped_ids(group_uids: np.ndarray) -> np.ndarray:
        return np.repeat(group_uids, group_size)

    def main_targets() -> torch.Tensor:
        rows = []
        for group_index in range(16):
            success_count = group_index % 7 + 1
            rows.extend([1.0] * success_count + [0.0] * (group_size - success_count))
        return torch.tensor(rows, dtype=torch.float32)

    def audit_targets(*, flipped: bool) -> torch.Tensor:
        rows = []
        for group_index in range(16):
            success_count = round(group_index * group_size / 15)
            group = [1.0] * success_count + [0.0] * (group_size - success_count)
            rows.extend([1.0 - value for value in group] if flipped else group)
        return torch.tensor(rows, dtype=torch.float32)

    def make_batch(group_uids: np.ndarray, targets: torch.Tensor) -> DataProto:
        batch_size = len(group_uids) * group_size
        responses = torch.ones((batch_size, 2), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, 4), dtype=torch.long)
        for row in range(batch_size):
            attention_mask[row, -(2 + row % 3) :] = 1
        token_rewards = torch.zeros((batch_size, 2), dtype=torch.float32)
        token_rewards[:, -1] = targets
        return DataProto.from_dict(
            tensors={
                "responses": responses,
                "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
                "attention_mask": attention_mask,
                "position_ids": torch.arange(4).expand(batch_size, -1).clone(),
                "step_end_mask": torch.tensor([[True, True]]).expand(batch_size, -1).clone(),
                "step_value_targets": targets.clone(),
                "step_value_task_rewards": targets.clone(),
                "token_level_rewards": token_rewards,
            },
            non_tensors={"uid": grouped_ids(group_uids)},
        )

    def run_once(*, flipped: bool):
        main = make_batch(main_group_uids, main_targets())
        audit = make_batch(audit_uids, audit_targets(flipped=flipped))
        trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
        trainer.global_steps = 1
        trainer.use_reference_policy = False
        trainer.use_prefix_grouper = False
        trainer.actor_rollout_wg = SimpleNamespace()
        trainer._get_dp_size = lambda *_: 24
        trainer.config = OmegaConf.create(
            {
                "trainer": {"balance_batch": True},
                "algorithm": {"adv_estimator": "step_value"},
                "actor_rollout_ref": {
                    "rollout": {"n": group_size},
                    "actor": {"loss_agg_mode": "token-mean", "loss_scale_factor": None},
                },
            }
        )
        trainer._prompt_center_audit_state = LaggedPromptCenterAuditState(
            initial_slope=1.0,
            initial_intercept=0.0,
            target_key="acc",
            group_size=group_size,
            audit_groups=16,
            rolling_window=2,
            seed=0,
        )
        trainer._pending_prompt_center_audit = None
        trainer._prepare_step_inputs = lambda _: None

        def prepare_provider_inputs(data: DataProto) -> None:
            data.meta_info["compute_step_value_probe"] = True
            data.meta_info["global_steps"] = trainer.global_steps

        trainer._prepare_step_value_inputs = prepare_provider_inputs
        captured: dict[str, torch.Tensor | tuple[float, float]] = {}
        main_size = len(main.batch)

        def fake_old_log_prob(data: DataProto):
            update_mask = data.batch["step_value_probe_update_mask"].detach().cpu().clone()
            audit_mask = data.batch["step_value_prompt_center_audit_mask"].detach().cpu().clone()
            row_ids = data.batch["step_value_forward_row_id"].detach().cpu().clone()
            captured["update_mask"] = update_mask
            captured["audit_mask"] = audit_mask
            captured["row_ids"] = row_ids
            captured["global_token_num"] = tuple(data.meta_info["global_token_num"])
            trajectory_means = torch.zeros(len(data.batch), dtype=torch.float32)
            audit_original_rows = row_ids[audit_mask] - main_size
            audit_ordinals = torch.div(audit_original_rows, group_size, rounding_mode="floor")
            trajectory_means[audit_mask] = -1.5 + 3.0 * audit_ordinals.float() / 15.0
            step_values = torch.full((len(data.batch), 2), 0.5)
            step_values[:, -1] = data.batch["step_value_targets"]
            output = DataProto.from_dict(
                tensors={
                    "old_log_probs": row_ids.float().unsqueeze(-1).expand(-1, 2).clone(),
                    "entropys": torch.ones((len(data.batch), 2)),
                    "step_values": step_values,
                    "step_value_trajectory_logit_mean": torch.zeros(len(data.batch)),
                    "step_value_ready": torch.ones(len(data.batch), dtype=torch.bool),
                    "step_value_probe_loss": torch.zeros(len(data.batch)),
                    "step_value_probe_grad_norm": torch.zeros(len(data.batch)),
                    "step_value_audit_trajectory_logit_mean": trajectory_means,
                    "step_value_audit_endpoint_count": 2 * audit_mask.long(),
                    "step_value_audit_ready_next": audit_mask,
                    "step_value_forward_row_id": row_ids,
                }
            )
            return output, 0.0

        trainer._compute_old_log_prob = fake_old_log_prob

        def complete_estimation(_: DataProto, *, prompt_center_parameters=None) -> None:
            captured["active_parameters"] = prompt_center_parameters

        trainer._complete_step_value_estimation = complete_estimation
        metrics: dict[str, float] = {}
        output = trainer.compute_kl_related_metrics_with_prompt_center_audit(
            main,
            audit,
            audit_uids,
            metrics,
            {},
        )
        return trainer, output, metrics, captured

    trainer, output, metrics, captured = run_once(flipped=False)
    flipped_trainer, flipped_output, _, _ = run_once(flipped=True)

    assert len(output.batch) == 16 * group_size
    assert captured["active_parameters"] == (1.0, 0.0)
    assert trainer._prompt_center_audit_state.active_parameters == (1.0, 0.0)
    assert trainer._pending_prompt_center_audit is not None
    assert len(trainer._pending_prompt_center_audit["centers"]) == 16
    np.testing.assert_array_equal(trainer._pending_prompt_center_audit["totals"], np.full(16, group_size))
    for key in (
        "step_value_audit_trajectory_logit_mean",
        "step_value_audit_endpoint_count",
        "step_value_audit_ready_next",
        "step_value_forward_row_id",
        "step_value_probe_update_mask",
        "step_value_prompt_center_audit_mask",
    ):
        assert key not in output.batch

    update_mask = captured["update_mask"]
    audit_mask = captured["audit_mask"]
    row_ids = captured["row_ids"]
    assert isinstance(update_mask, torch.Tensor)
    assert isinstance(audit_mask, torch.Tensor)
    assert isinstance(row_ids, torch.Tensor)
    assert int(update_mask.sum().item()) == 16 * group_size
    assert int(audit_mask.sum().item()) == 16 * group_size
    assert not bool((update_mask & audit_mask).any())
    assert int((~update_mask & ~audit_mask).sum().item()) == 8
    assert len(captured["global_token_num"]) == 2 * 16 * group_size
    assert sum(captured["global_token_num"]) == 2 * sum(2 + row % 3 for row in range(16 * group_size))
    assert not torch.equal(row_ids, torch.arange(row_ids.numel()))
    torch.testing.assert_close(output.batch["old_log_probs"][:, 0], torch.arange(16 * group_size).float())
    assert metrics["step_value/prompt_center_audit_reused_groups"] == 0.0
    assert metrics["step_value/prompt_center_audit_sidecar_groups"] == 16.0
    assert metrics["step_value/prompt_center_audit_neutral_padding_rows"] == 8.0
    assert metrics["step_value/prompt_center_audit_combined_balance_enabled"] == 1.0
    assert metrics["step_value/prompt_center_audit_all_zero_groups"] == 1.0
    assert metrics["step_value/prompt_center_audit_all_one_groups"] == 1.0
    assert metrics["step_value/prompt_center_audit_mixed_groups"] == 14.0
    assert "step_value/prompt_center_audit_combined_seqlen/balanced_max" in metrics
    assert metrics["step_value/prompt_center_audit_combined_rank_tokens_max"] >= metrics[
        "step_value/prompt_center_audit_combined_rank_tokens_min"
    ]

    initial_values, scales, active = prepare_step_value_context(
        output.batch["step_value_targets"],
        output.non_tensor_batch["uid"],
        norm_by_group_std=True,
        zero_when_group_uniform=False,
    )
    advantages, _ = compute_step_value_advantage(
        token_level_rewards=output.batch["token_level_rewards"],
        response_mask=output.batch["response_mask"],
        index=output.non_tensor_batch["uid"],
        step_values=output.batch["step_values"],
        step_end_mask=output.batch["step_end_mask"],
        step_value_initial_values=initial_values,
        step_value_scales=scales,
        step_value_active=active,
        step_value_task_rewards=output.batch["step_value_task_rewards"],
        step_value_ready=True,
        lam=0.9,
        norm_by_group_std=True,
    )
    flipped_advantages, _ = compute_step_value_advantage(
        token_level_rewards=flipped_output.batch["token_level_rewards"],
        response_mask=flipped_output.batch["response_mask"],
        index=flipped_output.non_tensor_batch["uid"],
        step_values=flipped_output.batch["step_values"],
        step_end_mask=flipped_output.batch["step_end_mask"],
        step_value_initial_values=initial_values,
        step_value_scales=scales,
        step_value_active=active,
        step_value_task_rewards=flipped_output.batch["step_value_task_rewards"],
        step_value_ready=True,
        lam=0.9,
        norm_by_group_std=True,
    )
    torch.testing.assert_close(advantages, flipped_advantages, atol=0.0, rtol=0.0)
    assert not np.array_equal(
        trainer._pending_prompt_center_audit["successes"],
        flipped_trainer._pending_prompt_center_audit["successes"],
    )

    trainer._commit_prompt_center_audit_after_advantage(metrics)
    flipped_trainer._commit_prompt_center_audit_after_advantage({})
    assert trainer._prompt_center_audit_state.active_parameters != (1.0, 0.0)
    assert trainer._prompt_center_audit_state.active_parameters != (
        flipped_trainer._prompt_center_audit_state.active_parameters
    )
