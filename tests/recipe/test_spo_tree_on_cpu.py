# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

import asyncio
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from recipe.dapo.spo_tree_core import (
    SPOTree,
    SPOTreeConfig,
    remaining_budget,
    spo_policy_loss,
    whiten_advantages,
)
from recipe.dapo.spo_tree_rollout import SPOTreeRollout, make_training_batch
from recipe.dapo.spo_tree_trainer import select_training_trees, tree_metrics
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.workers.config import FSDPActorConfig


def _tree(uid="q"):
    return SPOTree([1, 2], {"uid": uid, "data_source": "rlvr", "reward_model": {"ground_truth": "1"}})


def _config(max_response=8, max_model=None):
    return OmegaConf.create(
        {
            "algorithm": {"spo_tree": {"branching": [4, 2, 2], "segment_length": 2}},
            "data": {"max_response_length": max_response, "max_prompt_length": 4},
            "actor_rollout_ref": {
                "rollout": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_model_len": max_model,
                }
            },
            "reward": {"num_workers": 2},
        }
    )


class _Server:
    def __init__(self, early_stop=False, invalid=False):
        self.calls = []
        self.early_stop = early_stop
        self.invalid = invalid

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        index = len(self.calls)
        self.calls.append((list(prompt_ids), dict(sampling_params)))
        budget = sampling_params["max_tokens"]
        terminal = budget > 2 or (self.early_stop and index == 0)
        # Include EOS exactly at a length boundary for the early-stop case.
        ids = [10 + index] * budget
        if terminal:
            ids[-1] = 99
        return SimpleNamespace(
            token_ids=ids,
            stop_reason="aborted" if self.invalid else "completed",
            extra_fields={"finish_reason": "stop" if terminal else "length"},
        )


def _rollout(server, config=None, reward=None):
    return SPOTreeRollout(
        SimpleNamespace(pad_token_id=0, eos_token_id=99),
        config or _config(),
        server=server,
        reward_loop_manager=reward,
    )


def _generate(server=None, config=None):
    server = server or _Server()
    rollout = _rollout(server, config)
    trees = rollout.generate([_tree()], global_step=1, generation_batch=1)
    return trees[0], server


def _score_alternating(tree):
    for index, node in enumerate(tree.leaves):
        node.value = node.correctness = float(index % 2)
    tree.backpropagate_values()


def test_node_budget_and_invalid_config():
    config = SPOTreeConfig()
    assert config.max_nodes == 28
    assert config.max_leaves == 16
    assert SPOTreeConfig(branching=(4, 3, 2)).max_nodes == 40
    for kwargs in ({"branching": (4, 1, 2)}, {"branching": ()}, {"segment_length": 0}):
        with pytest.raises(ValueError):
            SPOTreeConfig(**kwargs)


def test_complete_422_tree_exact_token_prefix_and_budget():
    tree, server = _generate()
    assert Counter(node.depth for node in tree.nodes) == {0: 1, 1: 4, 2: 8, 3: 16}
    assert len(tree.leaves) == 16
    assert len(server.calls) == 28
    assert [params["max_tokens"] for _, params in server.calls] == [2] * 12 + [4] * 16
    assert len({params["seed"] for _, params in server.calls}) == 28
    for node, (prompt, params) in zip(tree.nodes[1:], server.calls, strict=True):
        assert prompt == tree.prompt_ids + tree.nodes[node.parent].response_ids
        assert node.response_ids[: node.start] == tree.nodes[node.parent].response_ids
        assert len(node.response_ids) <= 8
        assert params["return_finish_reason"] is True
        assert params["temperature"] == 1.0
    assert sum(len(node.segment_ids) for node in tree.nodes[1:]) == 4 * 2 + 8 * 2 + 16 * 4


def test_eos_exactly_at_cut_does_not_expand_or_duplicate_leaf():
    tree, server = _generate(_Server(early_stop=True))
    assert Counter(node.depth for node in tree.nodes) == {0: 1, 1: 4, 2: 6, 3: 12}
    assert len(tree.leaves) == 13
    assert not tree.nodes[1].children
    assert tree.nodes[1].finish_reason == "stop"
    for node in tree.leaves:
        node.value = node.correctness = float(node is tree.nodes[1])
    tree.backpropagate_values()
    assert tree.nodes[0].value == 0.25  # Not the flattened leaf average 1/13.
    assert tree.nodes[1].advantage == 0.75
    for leaf in tree.leaves:
        path_advantage = 0.0
        node = leaf
        while node.parent is not None:
            path_advantage += node.advantage
            node = tree.nodes[node.parent]
        assert path_advantage == pytest.approx(leaf.value - tree.nodes[0].value)


@pytest.mark.parametrize("config", [_config(max_response=2), _config(max_model=4)])
def test_context_or_response_exhaustion_never_requests_zero_tokens(config):
    tree, server = _generate(config=config)
    assert len(server.calls) == 4
    assert len(tree.leaves) == 4
    assert all(len(node.response_ids) == 2 for node in tree.leaves)
    assert all(params["max_tokens"] > 0 for _, params in server.calls)


def test_failed_generation_is_not_scored_as_a_terminal():
    with pytest.raises(RuntimeError, match="explicit finish_reason"):
        _generate(_Server(invalid=True))


def test_prefix_budget_excludes_question_but_respects_context():
    assert (
        remaining_budget(prompt_length=100, prefix_length=1200, max_response_length=10240, max_model_length=12288)
        == 9040
    )
    assert (
        remaining_budget(prompt_length=100, prefix_length=1200, max_response_length=10240, max_model_length=1500) == 200
    )
    with pytest.raises(ValueError):
        remaining_budget(prompt_length=10, prefix_length=11, max_response_length=10, max_model_length=20)


@pytest.mark.parametrize("early_stop", [False, True])
def test_node_training_masks_ancestors_and_keeps_zero_advantages(early_stop):
    tree, _ = _generate(_Server(early_stop=early_stop))
    _score_alternating(tree)
    node_count = 22 if early_stop else 28
    assert any(node.advantage == 0 for node in tree.nodes[1:])
    assert tree.training_nodes == tree.nodes[1:]
    assert len(tree.training_nodes) == node_count
    batch = make_training_batch([tree], SPOTreeConfig(segment_length=2), 0)
    assert len(batch) == 28
    assert batch.batch["spo_real_rows"].sum() == node_count
    for row, node in enumerate(tree.training_nodes):
        assert batch.batch["responses"][row, : len(node.response_ids)].tolist() == node.response_ids
        assert not batch.batch["response_mask"][row, : node.start].any()
        assert batch.batch["response_mask"][row].sum() == len(node.segment_ids)
        assert batch.batch["attention_mask"][row].sum() == len(tree.prompt_ids) + len(node.response_ids)
        assert torch.all(batch.batch["advantages"][row, node.start : len(node.response_ids)] == node.advantage)
    assert not batch.batch["response_mask"][node_count:].any()
    assert not batch.batch["advantages"][node_count:].any()
    assert torch.all(batch.batch["attention_mask"][node_count:].sum(dim=-1) == 2)
    metrics = tree_metrics([tree], max_response_length=8)
    assert metrics["spo/trainable_segments_per_prompt"] == node_count
    assert metrics["spo/segments_per_prompt"] == node_count


def test_dynamic_filter_depends_on_leaf_accuracy_not_node_advantages():
    tree, _ = _generate()
    # Correctness alone determines eligibility, even before advantages are populated.
    for index, node in enumerate(tree.leaves):
        node.correctness = float(index % 2)
    assert all(node.advantage == 0 for node in tree.nodes)
    assert select_training_trees([tree], "acc") == ([tree], 1)
    batch = make_training_batch([tree], SPOTreeConfig(segment_length=2), 0)
    assert batch.batch["spo_real_rows"].all()
    assert batch.batch["response_mask"].any(dim=-1).all()
    assert not batch.batch["advantages"].any()


@pytest.mark.parametrize("correctness", [0.0, 1.0])
@pytest.mark.parametrize("early_stop", [False, True])
def test_dynamic_filter_uses_accuracy_independently_of_length_penalties(correctness, early_stop):
    tree, _ = _generate(_Server(early_stop=early_stop))
    assert len(tree.leaves) == (13 if early_stop else 16)
    for node in tree.leaves:
        node.correctness = correctness
        node.value = 2 * correctness - 1
    tree.backpropagate_values()
    assert select_training_trees([tree], "acc") == ([], 0)
    assert select_training_trees([tree], "seq_reward") == ([], 0)
    tree.leaves[0].value -= 0.25  # Same correctness, different full-response length penalty.
    tree.backpropagate_values()
    assert select_training_trees([tree], "seq_reward") == ([tree], 1)
    assert select_training_trees([tree], "acc") == ([], 0)
    tree.leaves[-1].correctness = 1 - correctness
    tree.leaves[-1].value = 2 * (1 - correctness) - 1 - 0.5
    tree.backpropagate_values()
    values_before_filter = [node.value for node in tree.nodes]
    assert select_training_trees([tree], "acc") == ([tree], 1)
    assert [node.value for node in tree.nodes] == values_before_filter


def test_masked_whitening_matches_official_unbiased_token_statistics():
    values = torch.tensor([[0.2, 0.2, 9.0], [0.5, 0.5, 9.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    old = torch.tensor([[0.1, 0.95, 0.1], [0.2, 0.3, 0.1]]).log()
    output, active = whiten_advantages(values, mask, old)
    expected_active = torch.tensor([[True, False, False], [True, True, False]])
    assert torch.equal(active, expected_active)
    selected = values[expected_active]
    expected = (values - selected.mean()) / torch.sqrt(selected.var(unbiased=True) + 1e-8) * mask
    torch.testing.assert_close(output, expected)
    assert output[active].mean().abs() < 1e-6


def test_empty_probability_mask_falls_back_only_for_statistics():
    values = torch.tensor([[0.1, 0.9]])
    output, active = whiten_advantages(values, torch.ones_like(values).bool(), torch.full_like(values, 0.99).log())
    assert not active.any()
    assert torch.isfinite(output).all()
    assert output[0, 0] < 0 < output[0, 1]
    zeros, active = whiten_advantages(values, torch.zeros_like(values).bool(), torch.zeros_like(values))
    assert not zeros.any()
    assert not active.any()


def _actor_config():
    return FSDPActorConfig(
        strategy="fsdp",
        rollout_n=28,
        ppo_micro_batch_size_per_gpu=2,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        clip_ratio_c=10.0,
    )


def test_policy_loss_is_exactly_dapo_clip_higher_and_token_mean():
    config = _actor_config()
    old = torch.full((2, 3), 0.1).log()
    new = (old + torch.tensor([[0.4, 0.5, -0.9], [-0.5, 0.7, 0.0]])).requires_grad_()
    advantages = torch.tensor([[0.3, 0.3, 0.3], [-0.5, -0.5, -0.5]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    loss, _ = spo_policy_loss(new, old, advantages, mask, actor_config=config)
    expected, _ = compute_policy_loss_vanilla(old, new, advantages, mask, config=config)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert not new.grad[~mask].any()
    empty = torch.zeros_like(mask)
    zero_loss, _ = spo_policy_loss(new, old, advantages, empty, actor_config=config)
    assert zero_loss.item() == 0.0 and zero_loss.requires_grad


def test_reward_scoring_sees_full_response_and_removes_worker_padding():
    class Reward:
        def __init__(self):
            self.calls = []

        def compute_rm_score(self, batch):
            from verl import DataProto

            self.calls.append(batch)
            scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
            lengths = batch.batch["response_mask"].sum(dim=-1)
            for row, length in enumerate(lengths.tolist()):
                scores[row, length - 1] = 1.0 if length > 2 else 0.0
            return DataProto.from_dict(
                tensors={"rm_scores": scores},
                non_tensors={
                    "acc": np.asarray((lengths > 2).tolist(), dtype=object),
                },
            )

    tree, _ = _generate(_Server(early_stop=True))
    reward = Reward()
    _rollout(_Server(), reward=reward).score([tree])
    assert len(reward.calls[0]) == 14  # 13 true leaves, one temporary worker-padding row.
    assert tree.nodes[1].value == 0.0
    assert tree.nodes[0].value == 0.75
    assert len(tree.leaves) == 13


def test_batched_prompts_are_not_confused_or_weighted_by_leaf_count():
    server = _Server()
    trees = _rollout(server).generate([_tree("a"), _tree("b")], global_step=1, generation_batch=1)
    for tree in trees:
        _score_alternating(tree)
    batch = make_training_batch(trees, SPOTreeConfig(segment_length=2), 0)
    assert len(batch) == 56
    assert batch.non_tensor_batch["uid"].tolist() == ["a"] * 28 + ["b"] * 28
    assert all(len(tree.leaves) == 16 for tree in trees)


def test_checkpoint_epoch_and_settings_are_restored(tmp_path):
    from recipe.dapo.spo_tree_trainer import RaySPOTreeTrainer

    trainer = object.__new__(RaySPOTreeTrainer)
    trainer.spo_config = SPOTreeConfig()
    trainer.spo_epoch = 1
    trainer._save_trainer_extra_state(str(tmp_path))
    trainer.spo_epoch = 0
    trainer._load_trainer_extra_state(str(tmp_path))
    assert trainer.spo_epoch == 1
    trainer.spo_config = SPOTreeConfig(segment_length=800)
    with pytest.raises(ValueError, match="differs"):
        trainer._load_trainer_extra_state(str(tmp_path))


def test_script_arguments_and_hydra_config_without_submitting_ray():
    import subprocess

    from hydra import compose, initialize_config_dir

    from recipe.dapo.spo_tree_trainer import validate_spo_config
    from verl.experimental.reward_loop import migrate_legacy_reward_impl

    root = Path(__file__).resolve().parents[2]
    # Define a shell function named ray: the real CLI is never invoked.
    result = subprocess.run(
        ["bash", "-c", 'ray() { printf "%s\\n" "$@"; }; source recipe/dapo/spo_tree.sh'],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    args = result.stdout.splitlines()
    overrides = args[args.index("recipe.dapo.main_spo_tree") + 1 :]
    with initialize_config_dir(config_dir=str(root / "recipe/dapo/config"), version_base=None):
        defaults = compose(config_name="spo_tree_trainer")
        assert defaults.algorithm.filter_groups.metric == "acc"
        config = compose(config_name="spo_tree_trainer", overrides=overrides)
    config = migrate_legacy_reward_impl(config)
    spo = validate_spo_config(config)
    assert spo.branching == (4, 2, 2)
    assert config.actor_rollout_ref.rollout.n == 16
    assert config.actor_rollout_ref.actor.clip_ratio_high == 0.28
    assert not config.actor_rollout_ref.actor.use_kl_loss
    assert config.algorithm.filter_groups.metric == "acc"
    assert config.reward.reward_kwargs.overlong_buffer_cfg.enable
    assert config.reward.reward_kwargs.overlong_buffer_cfg.penalty_factor == 1.0
    assert config.data.train_batch_size == 128
    assert config.data.gen_batch_size == 384
    assert config.trainer.total_epochs == 2


@pytest.mark.parametrize("dynamic", [False, True])
def test_actual_actor_backward_keeps_prompt_optimizer_steps_and_frozen_old_policy(monkeypatch, dynamic):
    from dataclasses import replace

    from recipe.dapo.spo_tree_actor import SPODataParallelPPOActor

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8, vision_config=None)
            self.embedding = torch.nn.Embedding(128, 8)
            self.output = torch.nn.Linear(8, 128)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.output(self.embedding(input_ids)))

    def logprobs(logits, labels, **kwargs):
        return logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("verl.workers.actor.dp_actor.logprobs_from_logits", logprobs)
    monkeypatch.setattr("recipe.dapo.spo_tree_actor.get_device_id", lambda: torch.device("cpu"))
    config = replace(
        _actor_config(),
        ppo_mini_batch_size=28,  # One prompt/rank, independent of its real node count.
        ppo_epochs=1,
        use_dynamic_bsz=dynamic,
        ppo_max_token_len_per_gpu=80,
        use_remove_padding=False,
        use_torch_compile=False,
    )
    model = TinyLM()
    actor = SPODataParallelPPOActor(config, model, torch.optim.SGD(model.parameters(), lr=1e-2))
    actor.device_name = "cpu"
    actor.spo_config = SPOTreeConfig(segment_length=2)
    trees = _rollout(_Server()).generate([_tree("a"), _tree("b")], global_step=1, generation_batch=1)
    for tree in trees:
        _score_alternating(tree)
    batch = make_training_batch(trees, actor.spo_config, 0)
    batch.meta_info.update(temperature=1.0, pad_token_id=0)
    with torch.no_grad():
        old = actor._forward_micro_batch(dict(batch.batch), temperature=1.0, calculate_entropy=False)["log_probs"]
    batch.batch["old_log_probs"] = old.detach().clone()
    original_forward = actor._forward_micro_batch
    zero_advantage_rows_seen = []

    def track_forward(inputs, **kwargs):
        mask = inputs["response_mask"].bool()
        zero_rows = ((inputs["advantages"] == 0) | ~mask).all(dim=-1) & mask.any(dim=-1)
        zero_advantage_rows_seen.append(int(zero_rows.sum()))
        return original_forward(inputs, **kwargs)

    monkeypatch.setattr(actor, "_forward_micro_batch", track_forward)
    initial = model.output.weight.detach().clone()
    metrics = actor.update_policy(batch)
    assert sum(zero_advantage_rows_seen) == 24  # All 12 internal nodes per prompt reach the actor.
    assert metrics["spo/optimizer_steps"] == 2
    assert not torch.equal(initial, model.output.weight)
    assert all(np.isfinite(value) for value in metrics.values())
    assert "actor/kl_loss" not in metrics
    torch.testing.assert_close(batch.batch["old_log_probs"], old)


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("request_reason", [False, True])
def test_vllm_finish_reason_extension_is_opt_in_without_loading_engine(finish_reason, request_reason):
    # Execute the actual server method against an async fake engine. Importing
    # or constructing a vLLM engine is deliberately unnecessary for this test.
    import ast
    from typing import Any, Optional

    root = Path(__file__).resolve().parents[2]
    path = root / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    module = ast.parse(path.read_text())
    server_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "vLLMHttpServer"
    )
    method = next(
        node for node in server_class.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate"
    )
    method.decorator_list = []
    calls = []

    class Engine:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[9, 99], finish_reason=finish_reason)])

    def sampling_params(**kwargs):
        assert "return_finish_reason" not in kwargs  # Not passed to vLLM's SamplingParams.
        return SimpleNamespace(**kwargs)

    namespace = {
        "Optional": Optional,
        "Any": Any,
        "TokenOutput": SimpleNamespace,
        "RequestOutput": SimpleNamespace,
        "normalize_token_ids": list,
        "SamplingParams": sampling_params,
        "TokensPrompt": dict,
        "qwen2_5_vl_dedup_image_tokens": lambda ids, processor: ids,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    server = SimpleNamespace(
        config=OmegaConf.create({"max_model_len": 20, "enable_rollout_routing_replay": False}),
        model_config=SimpleNamespace(processor=None),
        engine=Engine(),
        lora_as_adapter=False,
        global_steps=7,
    )
    params = {"max_tokens": 2, "logprobs": False}
    if request_reason:
        params["return_finish_reason"] = True
    result = asyncio.run(namespace["generate"](server, prompt_ids=[1, 2], sampling_params=params, request_id="test"))
    assert result.stop_reason == "completed"
    assert result.extra_fields == {"global_steps": 7, **({"finish_reason": finish_reason} if request_reason else {})}
    assert calls[0]["sampling_params"].max_tokens == 2


@pytest.mark.parametrize("periodic_save, step_limit", [(False, 3), (True, 3), (False, 1)])
def test_dynamic_sampling_exit_validation_and_checkpoint_cursor(monkeypatch, tmp_path, periodic_save, step_limit):
    import json
    from copy import deepcopy

    from recipe.dapo.spo_tree_trainer import RaySPOTreeTrainer

    tree, _ = _generate()
    _score_alternating(tree)
    events, logs = [], []

    class Loader:
        cursor = 0

        def __iter__(self):
            for _ in range(3):
                self.cursor += 1
                yield {"input_ids": torch.ones(1, 1, dtype=torch.long)}

        def state_dict(self):
            return {"cursor": self.cursor}

    class Rollout:
        def prepare_prompts(self, batch):
            return [deepcopy(tree)]  # Only one passing prompt each round; two are needed.

        def generate(self, trees, **kwargs):
            events.append(("generate", kwargs["generation_batch"]))
            return trees

        def score(self, trees):
            pass

    trainer = object.__new__(RaySPOTreeTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": [],
                "total_epochs": 1,
                "val_before_train": True,
                "test_freq": 0,
                "save_freq": 1 if periodic_save else 0,
                "default_local_dir": str(tmp_path),
            },
            "algorithm": {"filter_groups": {"metric": "acc", "max_num_gen_batches": 10}},
            "data": {"train_batch_size": 2, "max_response_length": 8},
        }
    )
    trainer.spo_config = SPOTreeConfig(segment_length=2)
    trainer.spo_epoch = 0
    trainer.total_training_steps = step_limit
    trainer.train_dataloader = Loader()
    trainer.tokenizer = trainer.async_rollout_manager = trainer.reward_loop_manager = None
    trainer.actor_rollout_wg = SimpleNamespace()
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: events.append(("sleep", trainer.global_steps)),
        update_weights=lambda step: events.append(("weights", step)),
    )
    trainer._load_checkpoint = lambda: None

    def save():
        events.append(("save", trainer.global_steps))
        folder = tmp_path / f"global_step_{trainer.global_steps}"
        folder.mkdir(exist_ok=True)
        torch.save(trainer.train_dataloader.state_dict(), folder / "data.pt")
        trainer._save_trainer_extra_state(str(folder))

    def validate():
        events.append(("validate", trainer.global_steps))
        return {"val/acc": 0.5}

    def train(trees, timing):
        assert len(trees) == 2
        events.append(("update", trainer.global_steps))
        return {"actor/pg_loss": 0.1}

    trainer._save_checkpoint, trainer._validate, trainer._train_trees = save, validate, train
    monkeypatch.setattr("recipe.dapo.spo_tree_trainer.SPOTreeRollout", lambda *args: Rollout())
    monkeypatch.setattr(
        "verl.utils.tracking.Tracking", lambda **kwargs: SimpleNamespace(log=lambda **kw: logs.append(kw))
    )
    trainer.fit()
    assert trainer.global_steps == 1
    assert [event for event in events if event[0] == "validate"] == [("validate", 0), ("validate", 1)]
    assert [event for event in events if event[0] == "save"] == [("save", 1)]
    assert [event for event in events if event[0] == "update"] == [("update", 1)]
    train_log = next(log["data"] for log in logs if "train/num_gen_batches" in log["data"])
    assert train_log["train/num_gen_batches"] == 2
    assert train_log["train/num_prompt_pass_filter"] == 2
    state = json.loads((tmp_path / "global_step_1/spo_tree_state.json").read_text())
    assert state["epoch"] == (0 if step_limit == 1 else 1)
    cursor = torch.load(tmp_path / "global_step_1/data.pt", weights_only=True)
    assert cursor["cursor"] == (2 if step_limit == 1 else 3)


def test_step_timing_excludes_validation_but_includes_checkpoint(monkeypatch):
    from contextlib import contextmanager
    from unittest.mock import Mock

    import recipe.dapo.spo_tree_trainer as trainer_module

    tree, _ = _generate()
    _score_alternating(tree)
    elapsed = 0.0
    logs = []

    def consume(seconds, result=None):
        nonlocal elapsed
        elapsed += seconds
        return result

    @contextmanager
    def timer(name, timing, *args):
        started = elapsed
        yield
        timing[name] += elapsed - started

    monkeypatch.setattr(trainer_module, "perf_counter", lambda: elapsed)
    monkeypatch.setattr(trainer_module, "marked_timer", timer)
    monkeypatch.setattr(trainer_module, "tqdm", lambda **kwargs: Mock())
    monkeypatch.setattr(
        trainer_module,
        "SPOTreeRollout",
        lambda *args: SimpleNamespace(
            prepare_prompts=lambda batch: [tree],
            generate=lambda trees, **kwargs: consume(3, trees),
            score=lambda trees: consume(1),
        ),
    )
    monkeypatch.setattr(
        "verl.utils.tracking.Tracking", lambda **kwargs: SimpleNamespace(log=lambda **kw: logs.append(kw))
    )
    trainer = object.__new__(trainer_module.RaySPOTreeTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": [],
                "total_epochs": 1,
                "val_before_train": True,
                "test_freq": 10,
                "save_freq": 10,
            },
            "algorithm": {"filter_groups": {"metric": "acc", "max_num_gen_batches": 10}},
            "data": {"train_batch_size": 2, "max_response_length": 8},
        }
    )
    trainer.spo_epoch = 0
    trainer.total_training_steps = 11
    # Two generation batches per optimizer step; both must contribute to step time.
    trainer.train_dataloader = [{"input_ids": torch.ones(1, 1, dtype=torch.long)} for _ in range(22)]
    trainer.tokenizer = trainer.async_rollout_manager = trainer.reward_loop_manager = None
    trainer.actor_rollout_wg = SimpleNamespace()
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: None,
        update_weights=lambda step: consume(2),
    )
    trainer._load_checkpoint = lambda: None
    trainer._train_trees = lambda trees, timing: consume(20, {})
    trainer._save_checkpoint = Mock(side_effect=lambda: consume(40))
    trainer._validate = Mock(side_effect=lambda: consume(1000, {"val/acc": 0.5}))
    trainer._finish = Mock()  # Exit/checkpoint state is covered by the preceding test.
    trainer.fit()

    training_logs = [log["data"] for log in logs if "timing_s/step" in log["data"]]
    assert len(training_logs) == 11
    for step, metrics in enumerate(training_logs, start=1):
        assert metrics["train/num_gen_batches"] == 2
        assert metrics["timing_s/gen"] == 6
        assert metrics["timing_s/reward"] == 2
        assert metrics["timing_s/update_weights"] == 2
        if step in (10, 11):
            assert metrics["timing_s/save_checkpoint"] == 40
            assert metrics["timing_s/testing"] == 1000
            assert metrics["timing_s/step"] == 70
        else:
            assert "timing_s/testing" not in metrics
            assert "timing_s/save_checkpoint" not in metrics
            assert metrics["timing_s/step"] == 30
    assert trainer._validate.call_count == 3  # Initial, periodic, final.
    assert trainer._save_checkpoint.call_count == 2


def test_distributed_whitening_uses_all_ranks_token_statistics(monkeypatch):
    values = torch.tensor([[0.2, 0.4]])
    remote = torch.tensor([0.1, 0.6, 0.9])
    all_values = torch.cat([values.flatten(), remote])
    group_marker, calls = object(), []

    def all_reduce(tensor, *, group):
        assert group is group_marker
        if not calls:
            tensor.add_(torch.stack([remote.sum(), torch.tensor(float(remote.numel()))]))
        else:
            tensor.add_((remote - all_values.mean()).square().sum())
        calls.append(1)

    monkeypatch.setattr("recipe.dapo.spo_tree_core.dist.all_reduce", all_reduce)
    output, active = whiten_advantages(
        values,
        torch.ones_like(values).bool(),
        torch.full_like(values, 0.1).log(),
        distributed=True,
        group=group_marker,
    )
    expected = (values - all_values.mean()) / torch.sqrt(all_values.var(unbiased=True) + 1e-8)
    torch.testing.assert_close(output, expected)
    assert active.all() and len(calls) == 2
