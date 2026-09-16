# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Use the standard FSDP lifecycle with an SPO-specific policy updater."""

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .spo_tree_actor import SPODataParallelPPOActor
from .spo_tree_core import SPOTreeConfig


class SPOActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            # Reuse the SAME FSDP module, optimizer and checkpoint manager.
            # No monkey-patching of the base actor class or other recipes.
            actor = self.actor
            self.actor = SPODataParallelPPOActor(actor.config, actor.actor_module, actor.actor_optimizer)
            self.actor.spo_config = SPOTreeConfig(**dict(self.config.spo_tree))
