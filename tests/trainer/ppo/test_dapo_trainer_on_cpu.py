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

import pytest
from omegaconf import OmegaConf

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer


class _ExhaustedDataLoader:
    """A sized dataloader exhausted before the configured training step."""

    def __len__(self):
        return 1

    def __iter__(self):
        return iter(())


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
