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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
from verl import DataProto
from verl.trainer.ppo.sparse_counterfactual_credit import build_credit_residual, merge_anchor_credit


class _ExhaustedDataLoader:
    """A sized dataloader exhausted before the configured training step."""

    def __len__(self):
        return 1

    def __iter__(self):
        return iter(())


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("use_probe", [True, False])
def test_sparse_mc_disables_head_before_worker_initialization(enabled, use_probe):
    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"sparse_counterfactual_credit": {"enabled": enabled, "use_probe": use_probe}},
            "actor_rollout_ref": {
                "actor": {"counterfactual_credit_head": {"enabled": True}},
                "rollout": {"n": 8},
            },
        }
    )

    def check_head_config():
        assert trainer.config.actor_rollout_ref.actor.counterfactual_credit_head.enabled is (not enabled or use_probe)

    with patch("recipe.dapo.dapo_ray_trainer.RayPPOTrainer.init_workers", side_effect=check_head_config) as initialize:
        trainer.init_workers()
    initialize.assert_called_once_with()


@pytest.mark.parametrize("credit_enabled", [True, False])
@pytest.mark.parametrize("train_mc_branches", [True, False])
@pytest.mark.parametrize("use_probe", [True, False])
@pytest.mark.parametrize("credit_coef", [0.0, 0.3])
@pytest.mark.parametrize(("base_n", "branch_groups", "samples"), [(8, 2, 4), (6, 2, 3), (12, 3, 4)])
def test_actor_minibatch_response_count_includes_only_training_branches(
    credit_enabled, train_mc_branches, use_probe, credit_coef, base_n, branch_groups, samples
):
    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "sparse_counterfactual_credit": {
                    "enabled": credit_enabled,
                    "train_mc_branches": train_mc_branches,
                    "use_probe": use_probe,
                    "advantage_coef": credit_coef,
                    "branch_groups_per_prompt": branch_groups,
                    "num_samples": samples,
                }
            },
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 8,
                    "rollout_n": base_n,
                    "counterfactual_credit_head": {"enabled": True},
                },
                "rollout": {"n": base_n},
            },
        }
    )
    OmegaConf.set_struct(trainer.config, True)
    expected_n = base_n + (branch_groups * samples if credit_enabled and train_mc_branches else 0)

    def check_worker_config():
        assert trainer.config.actor_rollout_ref.rollout.n == base_n
        assert trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size == 8
        assert trainer.config.actor_rollout_ref.actor.rollout_n == expected_n

    with patch("recipe.dapo.dapo_ray_trainer.RayPPOTrainer.init_workers", side_effect=check_worker_config):
        trainer.init_workers()
        trainer.init_workers()  # Reinitialization must not multiply or add the branch count again.
    # A resolved config may already hold the previous training count.
    trainer.config.algorithm.sparse_counterfactual_credit.train_mc_branches = False
    with patch("recipe.dapo.dapo_ray_trainer.RayPPOTrainer.init_workers"):
        trainer.init_workers()
    assert trainer.config.actor_rollout_ref.actor.rollout_n == base_n
    assert trainer.config.actor_rollout_ref.rollout.n == base_n


@pytest.mark.parametrize("sign", [-1, 1])
def test_sparse_mc_skips_probe_and_centers_over_all_policy_tokens(sign):
    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create({"algorithm": {"sparse_counterfactual_credit": {"use_probe": False}}})
    trainer.actor_rollout_wg = MagicMock()
    anchor_mask = torch.tensor(
        [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    targets = torch.zeros(4, 6)
    targets[anchor_mask] = sign * torch.tensor([0.75, -0.5])
    response_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0]], dtype=torch.bool
    )
    batch = DataProto.from_dict(
        tensors={
            "credit_anchor_targets": targets,
            "credit_anchor_mask": anchor_mask,
            # Verify that stale dense values are never used in Sparse-MC mode.
            "credit_predictions": torch.ones_like(targets),
            "response_mask": response_mask,
            "step_end_mask": torch.tensor(
                [[0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 1, 0]], dtype=torch.bool
            ),
            "is_mc_branch": torch.tensor([False, False, False, True]),
        },
        non_tensors={"uid": np.asarray(["base", "base", "base", "base:mc-branch:1:0:q"], dtype=object)},
    )
    metrics, timing_raw = {}, {}
    result = trainer._predict_counterfactual_credit(batch, 0.3, metrics, timing_raw)

    assert result is batch
    trainer.actor_rollout_wg.update_counterfactual_credit.assert_not_called()
    assert not metrics and not timing_raw
    torch.testing.assert_close(result.batch["credit_predictions"], torch.zeros_like(targets))
    endpoints = merge_anchor_credit(result.batch["credit_predictions"], anchor_mask, targets[anchor_mask])
    torch.testing.assert_close(endpoints, targets)
    residual, _ = build_credit_residual(
        endpoints,
        batch.batch["step_end_mask"],
        response_mask,
        uids=batch.non_tensor_batch["uid"],
        epsilon=1e-6,
    )
    expected = sign * torch.tensor(
        [[0.5, 0.5, -0.25, -0.25, -0.25, -0.25], [0.25, 0.25, -0.25, -0.25, 0, 0], [0] * 6, [0] * 6]
    )
    # Base and generated branch tokens share the full-batch RMS.
    expected /= expected[response_mask].square().mean().sqrt() + 1e-6
    torch.testing.assert_close(residual, expected)
    torch.testing.assert_close(residual.sum(dim=-1), torch.zeros(4), atol=1e-6, rtol=0)
    # Non-anchor base rows, MC branches, prefixes, and padding all retain zero residual.
    assert not residual[2:].any()
    assert not residual[~response_mask].any()


@pytest.mark.parametrize("credit_coef", [0.0, 0.3])
@pytest.mark.parametrize("probe_config", [{}, {"use_probe": True}])
def test_full_method_keeps_probe_update_and_metrics(credit_coef, probe_config):
    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create(
        {"algorithm": {"sparse_counterfactual_credit": dict(
            probe_config, mc_value_loss_weight=1.0, start_value_loss_weight=0.0, terminal_value_loss_weight=0.0
        )}}
    )
    trainer.global_steps = 3
    trainer.actor_rollout_wg = MagicMock()
    predictions = torch.tensor([[0.2, -0.1]])
    values = {
        "credit_predictions": predictions,
        "credit_head_value_bce": torch.tensor([0.6]),
        "credit_head_value_mae": torch.tensor([0.1]),
        "credit_head_direction_agreement": torch.tensor([0.7]),
        "credit_head_direction_majority_baseline": torch.tensor([0.6]),
        "credit_head_direction_excess": torch.tensor([0.1]),
    }
    for source in ("mc", "start", "terminal"):
        for metric in ("bce", "mae"):
            values[f"credit_head_value_{metric}_{source}"] = torch.tensor([0.2 if source == "mc" else float("nan")])
    trainer.actor_rollout_wg.update_counterfactual_credit.return_value = DataProto.from_dict(tensors=values)
    batch = DataProto.from_dict(tensors={"credit_anchor_targets": torch.zeros_like(predictions)})
    metrics, timing_raw = {}, {}

    result = trainer._predict_counterfactual_credit(batch, credit_coef, metrics, timing_raw)

    trainer.actor_rollout_wg.update_counterfactual_credit.assert_called_once_with(batch)
    assert batch.meta_info["predict_all_credit_steps"] is (credit_coef != 0.0)
    assert batch.meta_info["credit_value_loss_weights"] == (1.0, 0.0, 0.0)
    assert batch.meta_info["global_steps"] == 3
    torch.testing.assert_close(result.batch["credit_predictions"], predictions)
    assert metrics["credit/head_direction_agreement"] == pytest.approx(0.7)
    assert metrics["credit/head_value_bce/mc"] == pytest.approx(0.2)
    assert "credit/head_value_bce/start" not in metrics
    assert "credit/head_value_mae/terminal" not in metrics
    assert set(metrics) == {
        "credit/head_value_bce", "credit/head_value_mae",
        "credit/head_value_bce/mc", "credit/head_value_mae/mc",
        "credit/head_direction_agreement", "credit/head_direction_majority_baseline",
        "credit/head_direction_excess",
    }
    assert "counterfactual_credit_head" in timing_raw


def test_response_lengths_use_full_base_responses_and_generated_mc_continuations():
    response_width = 6
    response_attention = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.zeros((4, response_width), dtype=torch.long),
            "attention_mask": torch.cat((torch.ones((4, 2), dtype=torch.long), response_attention), dim=-1),
            # This loss mask is deliberately non-canonical: generated length
            # must come from full attention length minus the stored prefix.
            "response_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0],
                    [0, 1, 1, 0, 0, 0],
                ],
                dtype=torch.bool,
            ),
            "is_mc_branch": torch.tensor([False, False, True, True]),
            "credit_prefix_lengths": torch.tensor([0, 0, 2, 1]),
        }
    )

    metrics = RayDAPOTrainer._compute_branch_response_length_metrics(
        batch,
        mc_branch_max_response_length=5,
    )

    assert metrics == {
        "response_length/base/mean": 5.0,
        "response_length/base/max": 6.0,
        "response_length/base/min": 4.0,
        "response_length/base/clip_ratio": 0.5,
        "response_length/mc_branch/mean": 2.5,
        "response_length/mc_branch/max": 3.0,
        "response_length/mc_branch/min": 2.0,
        "response_length/mc_branch/clip_ratio": 0.5,
    }


def test_counterfactual_credit_resumes_hybrid_rollout_via_weight_sync():
    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"checkpoint_engine": {"backend": "naive"}}}})
    trainer.global_steps = 7
    trainer.checkpoint_manager = MagicMock()

    trainer._resume_counterfactual_credit_rollout()

    trainer.checkpoint_manager.update_weights.assert_called_once_with(global_steps=7)
    trainer.checkpoint_manager.wake_up_replicas.assert_not_called()


def test_counterfactual_credit_rejects_backend_that_would_sleep_hybrid_twice():
    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"checkpoint_engine": {"backend": "nccl"}}}})
    trainer.global_steps = 7
    trainer.checkpoint_manager = MagicMock()

    with pytest.raises(ValueError, match="checkpoint_engine.backend=naive"):
        trainer._resume_counterfactual_credit_rollout()

    trainer.checkpoint_manager.update_weights.assert_not_called()


def test_dynamic_filter_shortfall_is_filled_in_input_order_from_eligible_filtered_prompts():
    selected, num_passed = RayDAPOTrainer._select_dynamic_filter_prompt_uids(
        {
            "passed": [0.0, 1.0],
            "filtered-first": [0.0, 0.0],
            "ineligible": [1.0, 1.0],
            "filtered-second": [1.0, 1.0],
        },
        branch_eligible_prompt_uids={"passed", "filtered-first", "filtered-second"},
        fill_shortfall=True,
        prompts_needed=3,
    )

    assert selected == ["passed", "filtered-first", "filtered-second"]
    assert num_passed == 1


def test_dynamic_filter_keeps_original_behavior_when_shortfall_fill_is_disabled():
    selected, num_passed = RayDAPOTrainer._select_dynamic_filter_prompt_uids(
        {"passed": [0.0, 1.0], "filtered": [0.0, 0.0]},
        branch_eligible_prompt_uids=None,
        fill_shortfall=False,
        prompts_needed=2,
    )

    assert selected == ["passed"]
    assert num_passed == 1


def _topup_test_batch(uids, metric_values, row_ids):
    return DataProto(
        batch=TensorDict({"row_id": torch.tensor(row_ids)}, batch_size=[len(row_ids)]),
        non_tensor_batch={
            "uid": np.asarray(uids, dtype=object),
            "acc": np.asarray(metric_values, dtype=np.float32),
        },
        meta_info={"source": "test"},
    )


def test_adaptive_topup_only_extends_homogeneous_groups_and_restores_initial_group_size():
    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "filter_groups": {
                    "adaptive_topup": {
                        "enabled": True,
                        "max_total_n": 8,
                        "chunk_size": 2,
                        "selection_seed": 7,
                    }
                }
            },
            "actor_rollout_ref": {"rollout": {"n": 4}},
        }
    )
    trainer.global_steps = 3
    prompt_uids = ["already-mixed", "rescued", "still-homogeneous"]
    prompt_context = _topup_test_batch(prompt_uids, [0, 0, 0], [100, 101, 102])
    initial_batch = _topup_test_batch(
        np.repeat(prompt_uids, 4),
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        list(range(12)),
    )
    calls = []

    def fake_generate_adaptive_topup(**kwargs):
        selected_uids = [prompt_uids[index] for index in kwargs["prompt_indices"]]
        calls.append(selected_uids)
        if len(calls) == 1:
            assert selected_uids == ["rescued", "still-homogeneous"]
            return _topup_test_batch(
                ["rescued", "rescued", "still-homogeneous", "still-homogeneous"],
                [0, 1, 1, 1],
                [20, 21, 22, 23],
            )
        assert selected_uids == ["still-homogeneous"]
        return _topup_test_batch(
            ["still-homogeneous", "still-homogeneous"],
            [1, 1],
            [24, 25],
        )

    trainer._generate_adaptive_topup = fake_generate_adaptive_topup
    result = trainer._adaptive_topup_dynamic_filter_groups(
        initial_batch,
        prompt_context=prompt_context,
        generation_context=prompt_context,
        metric_name="acc",
        metrics={},
        timing_raw={},
        topup_eligible_prompt_uids={"rescued", "still-homogeneous"},
        initial_anchor_eligible_rows=np.asarray(
            [False, False, False, False, True, True, False, False, True, True, False, False]
        ),
        required_anchor_rows=2,
    )

    assert calls == [["rescued", "still-homogeneous"], ["still-homogeneous"]]
    grouped_rows = {}
    grouped_metrics = {}
    for uid, metric, row_id in zip(
        result.non_tensor_batch["uid"],
        result.non_tensor_batch["acc"],
        result.batch["row_id"].tolist(),
        strict=True,
    ):
        grouped_rows.setdefault(uid, []).append(row_id)
        grouped_metrics.setdefault(uid, []).append(metric)

    assert all(len(rows) == 4 for rows in grouped_rows.values())
    assert grouped_rows["already-mixed"] == [0, 1, 2, 3]
    assert np.std(grouped_metrics["rescued"]) > 0
    assert any(row_id >= 20 for row_id in grouped_rows["rescued"])
    assert {4, 5}.issubset(grouped_rows["rescued"])
    assert grouped_rows["still-homogeneous"] == [8, 9, 10, 11]


def test_adaptive_topup_skips_groups_that_fail_independent_branch_eligibility():
    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "filter_groups": {
                    "adaptive_topup": {
                        "enabled": True,
                        "max_total_n": 4,
                        "chunk_size": 2,
                        "selection_seed": 7,
                    }
                }
            },
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )
    trainer.global_steps = 1
    prompt_uids = ["eligible", "ineligible"]
    prompt_context = _topup_test_batch(prompt_uids, [0, 0], [100, 101])
    initial_batch = _topup_test_batch(
        np.repeat(prompt_uids, 2),
        [0, 0, 1, 1],
        [0, 1, 2, 3],
    )

    def fake_generate_adaptive_topup(**kwargs):
        assert kwargs["prompt_indices"] == [0]
        return _topup_test_batch(["eligible", "eligible"], [0, 1], [10, 11])

    trainer._generate_adaptive_topup = fake_generate_adaptive_topup
    result = trainer._adaptive_topup_dynamic_filter_groups(
        initial_batch,
        prompt_context=prompt_context,
        generation_context=prompt_context,
        metric_name="acc",
        metrics={},
        timing_raw={},
        topup_eligible_prompt_uids={"eligible"},
    )

    grouped_metrics = RayDAPOTrainer._group_metric_values(result, "acc")
    assert np.std(grouped_metrics["eligible"]) > 0
    assert grouped_metrics["ineligible"] == [1, 1]


def test_dapo_validates_when_dataloader_is_exhausted_before_last_step(tmp_path):
    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": ["console"],
                "val_before_train": False,
                "total_epochs": 1,
                "test_freq": 1,
                "default_local_dir": str(tmp_path),
            },
            "actor_rollout_ref": {"rollout": {"skip_rollout": False}},
            "algorithm": {},
            "global_profiler": {"steps": None, "profile_continuous_steps": False},
        }
    )
    trainer.total_training_steps = 2
    trainer.train_dataloader = _ExhaustedDataLoader()
    trainer._load_checkpoint = MagicMock()
    trainer._validate = MagicMock(return_value={"val/test_score": 0.75})
    trainer.checkpoint_manager = MagicMock()

    logger = MagicMock()
    progress_bar = MagicMock()
    with (
        patch("verl.utils.tracking.Tracking", return_value=logger),
        patch("recipe.dapo.dapo_ray_trainer.tqdm", return_value=progress_bar),
        patch("recipe.dapo.dapo_ray_trainer.os.path.exists", return_value=True),
    ):
        trainer.fit()

    trainer._validate.assert_called_once_with()
    logged_data = logger.log.call_args.kwargs["data"]
    assert logged_data["val/test_score"] == 0.75
    assert "timing_s/testing" in logged_data
    assert "timing/testing" not in logged_data
    assert logger.log.call_args.kwargs["step"] == 1
    progress_bar.close.assert_called_once_with()


@pytest.mark.parametrize("consumed", [10, 68])
def test_legacy_checkpoint_resumes_remaining_data_and_second_epoch(tmp_path, consumed):
    from torchdata.stateful_dataloader import StatefulDataLoader

    original = StatefulDataLoader(list(range(68)), batch_size=1, num_workers=0)
    iterator = iter(original)
    for _ in range(consumed):
        next(iterator)
    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create({"trainer": {"total_epochs": 2}})
    trainer.global_steps = consumed
    trainer.train_dataloader = StatefulDataLoader(list(range(68)), batch_size=1, num_workers=0)
    trainer.train_dataloader.load_state_dict(original.state_dict())
    trainer._load_trainer_extra_state(str(tmp_path))
    batches = [(epoch, int(batch.item())) for epoch, batch in trainer._iter_training_batches()]
    assert batches == [(0, i) for i in range(consumed, 68)] + [(1, i) for i in range(68)]
    assert trainer._data_epoch == 2


@pytest.mark.parametrize("at_epoch_end", [False, True])
def test_data_epoch_checkpoint_is_independent_of_dynamic_sampling_update_count(tmp_path, at_epoch_end):
    from torchdata.stateful_dataloader import StatefulDataLoader

    trainer = RayDAPOTrainer.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create({"trainer": {"total_epochs": 3}})
    trainer._data_epoch = 1
    trainer.gen_steps = 99
    trainer.global_steps = 7  # Many data batches can correspond to few updates.
    trainer.train_dataloader = StatefulDataLoader(list(range(4)), batch_size=1, num_workers=0)
    iterator = iter(trainer.train_dataloader)
    consumed = 4 if at_epoch_end else 2
    for _ in range(consumed):
        next(iterator)
    loader_state = trainer.train_dataloader.state_dict()
    trainer._save_trainer_extra_state(str(tmp_path))
    restored = RayDAPOTrainer.__new__(RayDAPOTrainer)
    restored.config = trainer.config
    restored.global_steps = 7
    restored.train_dataloader = StatefulDataLoader(list(range(4)), batch_size=1, num_workers=0)
    restored.train_dataloader.load_state_dict(loader_state)
    restored._load_trainer_extra_state(str(tmp_path))
    assert restored.gen_steps == 99
    batches = [(epoch, int(batch.item())) for epoch, batch in restored._iter_training_batches()]
    assert batches == [(1, i) for i in range(consumed, 4)] + [(2, i) for i in range(4)]
