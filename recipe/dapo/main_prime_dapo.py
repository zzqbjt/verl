# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Entry point for DAPO with PRIME implicit process rewards."""

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device

from .main_dapo import DAPOTaskRunner


class PrimeDAPOTaskRunner(DAPOTaskRunner):
    """Register PRIME's online reward model alongside the DAPO workers."""

    def add_reward_model_resource_pool(self, config):
        if config.reward.reward_model.enable:
            raise ValueError("DAPO + PRIME expects reward.reward_model.enable=False")

        from recipe.prime.prime_fsdp_workers import PRIMERewardModelWorker

        self.role_worker_mapping[Role.RewardModel] = ray.remote(PRIMERewardModelWorker)
        self.mapping[Role.RewardModel] = "global_pool"

    def get_trainer_class(self):
        from .prime_dapo_ray_trainer import RayPrimeDAPOTrainer

        return RayPrimeDAPOTrainer


@hydra.main(config_path="config", config_name="prime_dapo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(PrimeDAPOTaskRunner))


if __name__ == "__main__":
    main()
