# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""CPU-testable SPO-Tree math and tree representation.

Reference: AIFrameResearch/SPO, tree_episode_generator.py, ppo_trainer.py
and trainers/utils.py. No probe, outcome-advantage mixture or replay buffer.
"""

from dataclasses import dataclass, field
from math import isfinite, prod
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class SPOTreeConfig:
    branching: tuple[int, ...] = (4, 2, 2)
    segment_length: int = 600
    # Official default: raw child-minus-parent differences, then whitening.
    normalize_sibling_std: bool = False
    probability_threshold: float = 0.9
    whiten_epsilon: float = 1e-8
    max_concurrent_requests: int = 128
    reward_batch_size: int = 256
    seed: int = 42

    def __post_init__(self):
        self.branching = tuple(self.branching)
        if not self.branching or any(type(n) is not int or n < 2 for n in self.branching):
            raise ValueError("SPO branching factors must be integers >= 2")
        if self.segment_length < 1:
            raise ValueError("SPO segment_length must be positive")
        if not 0 < self.probability_threshold <= 1:
            raise ValueError("SPO probability_threshold must be in (0, 1]")
        if self.whiten_epsilon <= 0:
            raise ValueError("SPO epsilon must be positive")
        if self.max_concurrent_requests < 1 or self.reward_batch_size < 1:
            raise ValueError("SPO concurrency and reward batch size must be positive")

    @property
    def max_leaves(self) -> int:
        return prod(self.branching)

    @property
    def max_nodes(self) -> int:
        return sum(prod(self.branching[:depth]) for depth in range(1, len(self.branching) + 1))


@dataclass
class SPONode:
    # Original question is separate; these are response tokens only.
    response_ids: list[int]
    start: int
    depth: int
    parent: int | None
    children: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    value: float | None = None
    correctness: float | None = None
    advantage: float = 0.0

    @property
    def segment_ids(self) -> list[int]:
        return self.response_ids[self.start :]


@dataclass
class SPOTree:
    prompt_ids: list[int]
    metadata: dict[str, Any]
    nodes: list[SPONode] = field(default_factory=lambda: [SPONode([], 0, 0, None)])

    @property
    def leaves(self) -> list[SPONode]:
        return [node for node in self.nodes[1:] if not node.children]

    @property
    def training_nodes(self) -> list[SPONode]:
        # Filter before whitening: whitening must not resurrect zero-credit nodes.
        return [node for node in self.nodes[1:] if node.advantage != 0.0]

    def backpropagate_values(self, normalize_sibling_std: bool = False) -> None:
        if len(self.nodes) < 2:
            raise ValueError("An SPO tree needs at least one generated node")
        for node in reversed(self.nodes):
            if node.children:
                values = [self.nodes[index].value for index in node.children]
                if any(value is None or not isfinite(value) for value in values):
                    raise ValueError("All children must have finite values before backup")
                # Average DIRECT children, not all leaves (different with early EOS).
                node.value = sum(values) / len(values)
            elif node.value is None or not isfinite(node.value):
                raise ValueError("Every terminal node must be scored before value backup")
        for node in self.nodes[1:]:
            parent = self.nodes[node.parent]
            node.advantage = node.value - parent.value
            if normalize_sibling_std:
                variance = sum((self.nodes[i].value - parent.value) ** 2 for i in parent.children)
                node.advantage /= (variance / len(parent.children)) ** 0.5 + 1e-8


def remaining_budget(*, prompt_length: int, prefix_length: int, max_response_length: int, max_model_length: int) -> int:
    if not 0 <= prefix_length <= max_response_length:
        raise ValueError("Response prefix is outside max_response_length")
    if prompt_length < 1 or prompt_length + prefix_length > max_model_length:
        raise ValueError("SPO prompt/prefix exceeds the model context")
    return min(max_response_length - prefix_length, max_model_length - prompt_length - prefix_length)


def probability_mask(response_mask: torch.Tensor, old_log_probs: torch.Tensor, threshold: float) -> torch.Tensor:
    if response_mask.shape != old_log_probs.shape:
        raise ValueError("SPO mask and old log probabilities must have matching shapes")
    return response_mask.bool() & (old_log_probs.float().exp() < threshold)


@torch.no_grad()
def whiten_advantages(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    *,
    threshold: float = 0.9,
    epsilon: float = 1e-8,
    distributed: bool = False,
    group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official masked whitening: subtract mean, divide by unbiased token std.

    Statistics are synchronized across the DP micro-batch, not per response or
    sibling pair. As upstream, a locally empty probability mask falls back to
    the segment mask for statistics only; the policy mask is NEVER relaxed.
    Empty/singleton global masks additionally get finite behavior.
    """
    if advantages.shape != response_mask.shape:
        raise ValueError("SPO advantages and response mask must have matching shapes")
    values = advantages.float()
    action_mask = probability_mask(response_mask, old_log_probs, threshold)
    stats_mask = action_mask if action_mask.any() else response_mask.bool()
    statistics = torch.stack(((values * stats_mask).sum(), stats_mask.sum().to(values.dtype)))
    if distributed:
        dist.all_reduce(statistics, group=group)
    total, count = statistics
    mean = total / count.clamp_min(1)
    squared_sum = ((values - mean).square() * stats_mask).sum()
    if distributed:
        dist.all_reduce(squared_sum, group=group)
    variance = squared_sum / (count - 1).clamp_min(1)
    whitened = (values - mean) * torch.rsqrt(variance + epsilon)
    return whitened * response_mask, action_mask


def spo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    actor_config,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use DAPO's existing token-level PPO loss, with SPO's probability mask.

    No separate SPO clipping implementation, reference KL, or ratio-skip rule.
    The empty-mask guard keeps layout-only FSDP micro-batches differentiable.
    """
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    if not action_mask.any():
        loss = log_probs.sum() * 0.0
        return loss, {"actor/pg_loss": 0.0, "actor/pg_clipfrac": 0.0, "actor/ppo_kl": 0.0}
    loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_probs,
        log_prob=log_probs,
        advantages=advantages,
        response_mask=action_mask,
        loss_agg_mode="token-mean",
        config=actor_config,
    )
    metrics["actor/pg_loss"] = float(loss.detach())
    return loss, metrics
