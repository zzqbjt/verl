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

import os

import pytest
from hydra import compose, initialize_config_dir

from verl.trainer.config import SparseCounterfactualCreditConfig
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import CounterfactualCreditHeadConfig


def test_sparse_credit_defaults_match_method_configuration():
    config = SparseCounterfactualCreditConfig()
    assert not config.enabled
    assert config.use_probe
    assert config.entropy_top_ratio == 0.2
    assert config.anchors_per_group == 2
    assert config.num_samples == 4
    assert not config.train_mc_branches
    assert config.branch_groups_per_prompt == 2
    assert config.correctness_key == "acc"
    assert config.mc_value_loss_weight == 0.6
    assert config.start_value_loss_weight == 0.2
    assert config.terminal_value_loss_weight == 0.2
    assert config.advantage_coef == 0.3
    assert config.warmup_ratio == 0.1
    assert config.max_new_tokens is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"use_probe": 1}, "use_probe"),
        ({"use_probe": "False"}, "use_probe"),
        ({"entropy_top_ratio": 0.0}, "entropy_top_ratio"),
        ({"anchors_per_group": 0}, "anchors_per_group"),
        ({"sampling_temperature": 0.0}, "sampling_temperature"),
        ({"uniform_mix": 1.1}, "uniform_mix"),
        ({"num_samples": 0}, "num_samples"),
        ({"num_samples": -1}, "num_samples"),
        ({"num_samples": 1.5}, "num_samples"),
        ({"num_samples": True}, "num_samples"),
        ({"train_mc_branches": 1}, "train_mc_branches"),
        ({"branch_groups_per_prompt": 0}, "branch_groups_per_prompt"),
        ({"correctness_key": ""}, "correctness_key"),
        ({"mc_value_loss_weight": -0.1}, "mc_value_loss_weight"),
        (
            {
                "mc_value_loss_weight": 0.0,
                "start_value_loss_weight": 0.0,
                "terminal_value_loss_weight": 0.0,
            },
            "may not all be zero",
        ),
        ({"advantage_coef": 1.1}, "advantage_coef"),
        ({"top_k": 0}, "top_k"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
    ],
)
def test_sparse_credit_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SparseCounterfactualCreditConfig(**kwargs)


def test_credit_head_defaults_and_validation():
    config = CounterfactualCreditHeadConfig()
    assert not config.enabled
    assert config.hidden_dim == 512
    assert config.lr == 1e-3
    assert config.optimizer_steps_per_batch == 2
    assert config.save_checkpoint
    with pytest.raises(ValueError, match="hidden_dim"):
        CounterfactualCreditHeadConfig(hidden_dim=0)
    with pytest.raises(ValueError, match="optimizer_steps_per_batch"):
        CounterfactualCreditHeadConfig(optimizer_steps_per_batch=0)


@pytest.mark.parametrize("use_probe", [True, False])
def test_ppo_yaml_materializes_both_typed_configs(use_probe):
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                f"algorithm.sparse_counterfactual_credit.use_probe={use_probe}",
                "algorithm.sparse_counterfactual_credit.num_samples=3",
            ],
        )
    algorithm = omega_conf_to_dataclass(config.algorithm)
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert isinstance(algorithm.sparse_counterfactual_credit, SparseCounterfactualCreditConfig)
    assert algorithm.sparse_counterfactual_credit.use_probe is use_probe
    assert algorithm.sparse_counterfactual_credit.num_samples == 3
    assert "num_q_samples" not in config.algorithm.sparse_counterfactual_credit
    assert "num_v_samples" not in config.algorithm.sparse_counterfactual_credit
    assert isinstance(actor.counterfactual_credit_head, CounterfactualCreditHeadConfig)
