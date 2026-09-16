# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""SPO-Tree (4-2-2) with DAPO prompt-level dynamic sampling."""

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device

from .main_dapo import DAPOTaskRunner


class SPOTreeTaskRunner(DAPOTaskRunner):
    def add_actor_rollout_worker(self, config):
        from .spo_tree_trainer import validate_spo_config
        from .spo_tree_workers import SPOActorRolloutRefWorker

        validate_spo_config(config)
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(SPOActorRolloutRefWorker)
        self.mapping[Role.ActorRollout] = "global_pool"
        return SPOActorRolloutRefWorker, RayWorkerGroup

    def get_trainer_class(self):
        from .spo_tree_trainer import RaySPOTreeTrainer

        return RaySPOTreeTrainer


@hydra.main(config_path="config", config_name="spo_tree_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SPOTreeTaskRunner))


if __name__ == "__main__":
    main()
