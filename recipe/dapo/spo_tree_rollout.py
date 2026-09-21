# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Exact-token SPO tree generation, terminal scoring, and segment batching."""

import asyncio
from copy import deepcopy
from uuid import uuid4

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.chat_template import apply_chat_template
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import auto_await
from verl.utils.tokenizer import normalize_token_ids

from .spo_tree_core import SPONode, SPOTree, SPOTreeConfig, remaining_budget


def make_token_batch(
    examples: list[tuple[SPOTree, SPONode | None]], pad_token_id: int, *, training: bool = False
) -> DataProto:
    """Keep full response prefixes in attention, but never in segment loss.

    None denotes a layout-only dummy. Its two attended tokens keep FSDP
    collectives safe, but it has no loss, reward, or statistical weight.
    """
    if not examples:
        raise ValueError("Cannot construct an empty SPO batch")
    prompt_width = max(len(tree.prompt_ids) if node is not None else 1 for tree, node in examples)
    response_width = max(len(node.response_ids) if node is not None else 1 for _, node in examples)
    rows = len(examples)
    prompts = torch.full((rows, prompt_width), pad_token_id, dtype=torch.long)
    responses = torch.full((rows, response_width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((rows, prompt_width + response_width), dtype=torch.long)
    response_mask = torch.zeros((rows, response_width), dtype=torch.bool)
    advantages = torch.zeros((rows, response_width), dtype=torch.float32)
    scores = torch.zeros_like(advantages)
    real_rows = torch.zeros(rows, dtype=torch.bool)
    prefix_lengths = torch.zeros(rows, dtype=torch.long)
    metadata_keys = ("uid", "data_source", "reward_model", "extra_info")
    metadata = {key: np.empty(rows, dtype=object) for key in metadata_keys}
    for row, (tree, node) in enumerate(examples):
        for key in metadata_keys:
            metadata[key][row] = deepcopy(tree.metadata.get(key, {} if key == "extra_info" else None))
        if node is None:
            prompts[row, -1] = tree.prompt_ids[-1]
            attention_mask[row, prompt_width - 1 : prompt_width + 1] = 1
            continue
        if not node.response_ids or not 0 <= node.start < len(node.response_ids):
            raise ValueError("SPO nodes require a nonempty generated segment")
        prompt_len, response_len = len(tree.prompt_ids), len(node.response_ids)
        prompts[row, -prompt_len:] = torch.tensor(tree.prompt_ids)
        responses[row, :response_len] = torch.tensor(node.response_ids)
        attention_mask[row, prompt_width - prompt_len : prompt_width + response_len] = 1
        start = node.start if training else 0
        response_mask[row, start:response_len] = True
        advantages[row, start:response_len] = node.advantage
        if node.value is not None:
            scores[row, response_len - 1] = node.value
        real_rows[row] = True
        prefix_lengths[row] = start
    tensors = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": torch.cat((prompts, responses), dim=-1),
        "attention_mask": attention_mask,
        "position_ids": compute_position_id_with_mask(attention_mask),
        "response_mask": response_mask,
    }
    if training:
        tensors.update(
            advantages=advantages,
            returns=advantages.clone(),
            token_level_scores=scores,
            token_level_rewards=scores.clone(),
            spo_real_rows=real_rows,
            spo_prefix_lengths=prefix_lengths,
        )
    return DataProto.from_dict(tensors=tensors, non_tensors=metadata)


def make_training_batch(trees: list[SPOTree], config: SPOTreeConfig, pad_token_id: int) -> DataProto:
    """Reserve max_nodes slots per prompt to preserve prompt mini-batch sizes.

    The actor discards dummy slots before loss/whitening. These slots are NOT
    repeated trajectories and do not add optimizer steps or training weight.
    """
    examples = []
    for tree in trees:
        nodes = tree.training_nodes
        if not nodes:
            raise ValueError("An SPO training prompt must contain a generated segment")
        if len(tree.nodes) - 1 > config.max_nodes:
            raise ValueError("Tree exceeds its configured node budget")
        examples.extend((tree, node) for node in nodes)
        examples.extend((tree, None) for _ in range(config.max_nodes - len(nodes)))
    return make_token_batch(examples, pad_token_id, training=True)


class SPOTreeRollout:
    def __init__(self, tokenizer, trainer_config, rollout_manager=None, reward_loop_manager=None, *, server=None):
        self.tokenizer = tokenizer
        self.trainer_config = trainer_config
        self.config = SPOTreeConfig(**dict(trainer_config.algorithm.spo_tree))
        self.rollout_config = trainer_config.actor_rollout_ref.rollout
        self.reward_loop_manager = reward_loop_manager
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("SPO requires a pad or EOS token id")
        self.max_response_length = int(trainer_config.data.max_response_length)
        self.max_model_length = self.rollout_config.get("max_model_len") or (
            int(trainer_config.data.max_prompt_length) + self.max_response_length
        )
        if server is None:
            from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

            server = AsyncLLMServerManager(
                config=trainer_config,
                servers=list(zip(rollout_manager.server_addresses, rollout_manager.server_handles, strict=True)),
                load_balancer_handle=rollout_manager.global_load_balancer,
            )
        self.server = server

    def prepare_prompts(self, batch: DataProto) -> list[SPOTree]:
        trees = []
        template_kwargs = dict(self.trainer_config.data.get("apply_chat_template_kwargs", {}))
        for key in ("tokenize", "return_tensors", "return_dict", "add_generation_prompt"):
            template_kwargs.pop(key, None)
        for row in range(len(batch)):
            metadata = {key: deepcopy(values[row]) for key, values in batch.non_tensor_batch.items()}
            messages = list(metadata["raw_prompt"])
            if any(not isinstance(message.get("content"), str) for message in messages):
                raise ValueError("SPO-Tree currently supports text-only prompts")
            ids = normalize_token_ids(
                apply_chat_template(
                    self.tokenizer, messages, tools=None, add_generation_prompt=True, tokenize=True, **template_kwargs
                )
            )
            if not ids or len(ids) > int(self.trainer_config.data.max_prompt_length):
                raise ValueError("SPO prompt exceeds max_prompt_length; enable filter_overlong_prompts")
            metadata["uid"] = uuid4().hex
            trees.append(SPOTree(ids, metadata))
        return trees

    @auto_await
    async def generate(self, trees: list[SPOTree], *, global_step: int, generation_batch: int) -> list[SPOTree]:
        """Generate each edge exactly once; a stopped branch is never expanded."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        frontier = [(tree, 0) for tree in trees]
        request_index = 0
        seed_base = int(np.random.SeedSequence([self.config.seed, global_step, generation_batch]).generate_state(1)[0])

        async def sample(tree, parent_index, budget, seed):
            parent = tree.nodes[parent_index]
            params = {
                "max_tokens": budget,
                "temperature": float(self.rollout_config.temperature),
                "top_p": float(self.rollout_config.top_p),
                "top_k": int(self.rollout_config.top_k),
                "repetition_penalty": float(self.rollout_config.get("repetition_penalty", 1.0)),
                "logprobs": False,
                "return_finish_reason": True,
                "seed": seed,
            }
            async with semaphore:
                output = await self.server.generate(
                    request_id=f"spo-tree-{tree.metadata['uid']}",
                    prompt_ids=tree.prompt_ids + parent.response_ids,
                    sampling_params=params,
                )
            ids = normalize_token_ids(output.token_ids)
            reason = output.extra_fields.get("finish_reason")
            if output.stop_reason == "aborted" or reason not in {"length", "stop"}:
                raise RuntimeError(f"SPO needs a completed rollout with an explicit finish_reason, got {reason!r}")
            if not ids or len(ids) > budget:
                raise RuntimeError(f"SPO generated {len(ids)} tokens for a {budget}-token request")
            if reason == "length" and len(ids) != budget:
                raise RuntimeError("SPO rollout ended before its length budget without a terminal stop")
            return ids, reason

        for depth, branch_factor in enumerate(self.config.branching, start=1):
            requests, tasks = [], []
            for tree, parent_index in frontier:
                parent = tree.nodes[parent_index]
                budget = remaining_budget(
                    prompt_length=len(tree.prompt_ids),
                    prefix_length=len(parent.response_ids),
                    max_response_length=self.max_response_length,
                    max_model_length=self.max_model_length,
                )
                if not budget:
                    parent.finish_reason = "length"
                    continue
                if depth < len(self.config.branching):
                    budget = min(budget, self.config.segment_length)
                for _ in range(branch_factor):
                    seed = (seed_base + request_index) % (2**31 - 1)
                    request_index += 1
                    requests.append((tree, parent_index))
                    tasks.append(asyncio.create_task(sample(tree, parent_index, budget, seed)))
            try:
                outputs = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            frontier = []
            for (tree, parent_index), (ids, reason) in zip(requests, outputs, strict=True):
                parent = tree.nodes[parent_index]
                node_index = len(tree.nodes)
                tree.nodes.append(
                    SPONode(
                        parent.response_ids + ids, len(parent.response_ids), depth, parent_index, finish_reason=reason
                    )
                )
                parent.children.append(node_index)
                if reason == "length" and depth < len(self.config.branching):
                    frontier.append((tree, node_index))
            if not frontier:
                break
        return trees

    def score(self, trees: list[SPOTree]) -> None:
        examples = [(tree, node) for tree in trees for node in tree.leaves]
        for start in range(0, len(examples), self.config.reward_batch_size):
            chunk = examples[start : start + self.config.reward_batch_size]
            batch = make_token_batch(chunk, self.pad_token_id)
            padded, padding = pad_dataproto_to_divisor(batch, int(self.trainer_config.reward.num_workers))
            output = unpad_dataproto(self.reward_loop_manager.compute_rm_score(padded), padding)
            rewards = output.batch["rm_scores"].sum(dim=-1).float().cpu()
            if "acc" not in output.non_tensor_batch:
                raise ValueError("SPO dynamic sampling requires the verifier's acc field")
            correctness = np.asarray(output.non_tensor_batch["acc"], dtype=np.float64)
            if not torch.isfinite(rewards).all() or not np.isfinite(correctness).all():
                raise ValueError("SPO terminal rewards must be finite")
            if np.any((correctness < 0) | (correctness > 1)):
                raise ValueError("SPO correctness must be in [0, 1]")
            for (_, node), reward, acc in zip(chunk, rewards.tolist(), correctness.tolist(), strict=True):
                # The verifier sees question + FULL response; length penalties
                # apply to prefix + suffix, not to each individual 600-token edge.
                node.value, node.correctness = reward, acc
        for tree in trees:
            tree.backpropagate_values(self.config.normalize_sibling_std)
