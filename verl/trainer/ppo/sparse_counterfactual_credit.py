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
"""Pure tensor operations for sparse counterfactual step credit.

The rollout and actor RPC orchestration intentionally lives outside this
module.  Keeping uncertainty, sampling, target construction, and advantage
residuals here makes their invariants independently CPU-testable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _validate_step_tensors(
    values: torch.Tensor,
    step_end_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or step_end_mask.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("values, step_end_mask, and response_mask must be rank-2")
    if values.shape != step_end_mask.shape or values.shape != response_mask.shape:
        raise ValueError("values, step_end_mask, and response_mask must have identical shapes")
    ends = step_end_mask.bool()
    active = response_mask.bool()
    if torch.any(ends & ~active):
        raise ValueError("step_end_mask may select only active response tokens")
    if torch.any(ends.sum(dim=-1) == 0):
        raise ValueError("every response must contain at least one step endpoint")
    return ends, active


def compute_step_uncertainty(
    token_entropy: torch.Tensor,
    step_end_mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    top_ratio: float = 0.2,
) -> torch.Tensor:
    """Place each step's top-token entropy mean at its semantic endpoint."""

    if not 0.0 < top_ratio <= 1.0:
        raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}")
    ends, active = _validate_step_tensors(token_entropy, step_end_mask, response_mask)
    uncertainty = torch.zeros_like(token_entropy)
    for row in range(token_entropy.shape[0]):
        active_positions = torch.nonzero(active[row], as_tuple=False).flatten()
        end_positions = torch.nonzero(ends[row], as_tuple=False).flatten()
        active_rank = {int(position): rank for rank, position in enumerate(active_positions.tolist())}
        start_rank = 0
        for end_position_tensor in end_positions:
            end_position = int(end_position_tensor)
            end_rank = active_rank[end_position]
            if end_rank < start_rank:
                raise ValueError(f"response {row}: endpoints do not define nonempty ordered steps")
            span = active_positions[start_rank : end_rank + 1]
            top_count = max(1, math.ceil(span.numel() * top_ratio))
            uncertainty[row, end_position] = torch.topk(token_entropy[row, span].float(), top_count).values.mean()
            start_rank = end_rank + 1
    return uncertainty


def sample_anchor_steps(
    step_uncertainty: torch.Tensor,
    step_end_mask: torch.Tensor,
    uids: Sequence[object],
    *,
    temperature: float = 1.0,
    uniform_mix: float = 0.1,
    anchors_per_group: int = 2,
    seed: int = 42,
    global_step: int = 0,
) -> torch.Tensor:
    """Sequentially sample endpoints from distinct responses in each uid group.

    After every draw, all endpoints from the selected response are removed.
    Entropy and uniform-mixture weights are recomputed over the remaining
    endpoints before every draw, so a response can contribute at most one
    anchor to a group.
    """

    if step_uncertainty.ndim != 2 or step_end_mask.shape != step_uncertainty.shape:
        raise ValueError("step_uncertainty and step_end_mask must be identically-shaped rank-2 tensors")
    if len(uids) != step_uncertainty.shape[0]:
        raise ValueError("uids length must equal the response batch size")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and > 0")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must be in [0, 1]")
    if not isinstance(anchors_per_group, int) or isinstance(anchors_per_group, bool) or anchors_per_group < 1:
        raise ValueError("anchors_per_group must be an integer >= 1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(global_step))
    anchor_mask = torch.zeros_like(step_end_mask, dtype=torch.bool)
    groups: dict[object, list[int]] = {}
    for row, uid in enumerate(uids):
        groups.setdefault(uid, []).append(row)
    for rows in groups.values():
        available = step_end_mask[rows].detach().cpu().bool().clone()
        eligible_responses = int(available.any(dim=-1).sum().item())
        if eligible_responses < anchors_per_group:
            raise ValueError(
                f"uid group has {eligible_responses} responses with step endpoints, "
                f"fewer than anchors_per_group={anchors_per_group}"
            )
        group_uncertainty = step_uncertainty[rows].detach().cpu().float()
        for _ in range(anchors_per_group):
            coordinates = torch.nonzero(available, as_tuple=False)
            scores = group_uncertainty[coordinates[:, 0], coordinates[:, 1]]
            entropy_probability = torch.softmax(scores / temperature, dim=0)
            weights = (1.0 - uniform_mix) * entropy_probability + uniform_mix / scores.numel()
            if torch.any(~torch.isfinite(weights)) or torch.any(weights <= 0):
                raise ValueError("all available endpoint sampling probabilities must be finite and positive")
            selected = int(torch.multinomial(weights, 1, replacement=False, generator=generator).item())
            local_row, endpoint = coordinates[selected].tolist()
            anchor_mask[rows[local_row], endpoint] = True
            available[local_row] = False
    return anchor_mask


def compute_monte_carlo_credit(q_rewards: torch.Tensor, v_rewards: torch.Tensor) -> torch.Tensor:
    """Compute signed anchor targets from exact Q and V sample matrices."""

    if q_rewards.ndim != 2 or v_rewards.ndim != 2:
        raise ValueError("q_rewards and v_rewards must be rank-2")
    if q_rewards.shape[0] != v_rewards.shape[0] or q_rewards.shape[1] < 1 or v_rewards.shape[1] < 1:
        raise ValueError("Q and V rewards must contain the same nonempty anchor dimension")
    return q_rewards.float().mean(dim=-1) - v_rewards.float().mean(dim=-1)


def merge_anchor_credit(
    predicted_credit: torch.Tensor,
    anchor_mask: torch.Tensor,
    anchor_credit: torch.Tensor,
) -> torch.Tensor:
    """Use Monte-Carlo labels at anchors and pre-update predictions elsewhere."""

    if predicted_credit.shape != anchor_mask.shape or predicted_credit.ndim != 2:
        raise ValueError("predicted_credit and anchor_mask must be identically-shaped rank-2 tensors")
    if anchor_credit.ndim != 1 or anchor_credit.numel() != int(anchor_mask.sum().item()):
        raise ValueError("anchor_credit must contain one value per selected anchor")
    merged = predicted_credit.clone()
    merged[anchor_mask.bool()] = anchor_credit.to(device=merged.device, dtype=merged.dtype)
    return merged


def expand_step_credit_to_tokens(
    endpoint_credit: torch.Tensor,
    step_end_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcast endpoint credit over policy tokens, including trailing final EOS actions."""

    ends, active = _validate_step_tensors(endpoint_credit, step_end_mask, response_mask)
    token_credit = torch.zeros_like(endpoint_credit)
    for row in range(endpoint_credit.shape[0]):
        active_positions = torch.nonzero(active[row], as_tuple=False).flatten()
        end_positions = torch.nonzero(ends[row], as_tuple=False).flatten().tolist()
        active_rank = {int(position): rank for rank, position in enumerate(active_positions.tolist())}
        start_rank = 0
        for step_index, end_position in enumerate(end_positions):
            end_rank = active_rank[end_position]
            is_last_step = step_index == len(end_positions) - 1
            stop_rank = active_positions.numel() if is_last_step else end_rank + 1
            span = active_positions[start_rank:stop_rank]
            token_credit[row, span] = endpoint_credit[row, end_position]
            start_rank = end_rank + 1
    return token_credit


def build_credit_residual(
    endpoint_credit: torch.Tensor,
    step_end_mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    normalize_batch_std: bool = True,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-weight center per response, then apply token-weighted batch scaling."""

    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    active = response_mask.bool()
    residual = expand_step_credit_to_tokens(endpoint_credit, step_end_mask, response_mask)
    counts = active.sum(dim=-1).clamp_min(1)
    means = (residual * active).sum(dim=-1) / counts
    residual = (residual - means.unsqueeze(-1)) * active
    scale = residual.new_tensor(1.0)
    if normalize_batch_std:
        scale = torch.sqrt(residual.square().sum() / active.sum().clamp_min(1))
        residual = residual / (scale + epsilon)
    metrics = {
        "credit/residual_abs_mean": float(residual[active].abs().mean().item()),
        "credit/residual_scale": float(scale.item()),
    }
    return residual, metrics


def credit_advantage_coefficient(
    *,
    global_step: int,
    total_training_steps: int,
    maximum: float,
    warmup_ratio: float,
) -> float:
    """Linear warmup whose first optimizer update receives zero credit residual."""

    if not 0.0 <= maximum <= 1.0 or not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("maximum and warmup_ratio must be in [0, 1]")
    if warmup_ratio == 0.0:
        return float(maximum)
    if total_training_steps < 1:
        raise ValueError("total_training_steps must be >= 1 when warmup_ratio > 0")
    if global_step <= 1:
        return 0.0
    warmup_steps = max(1, math.ceil(total_training_steps * warmup_ratio))
    progress = min(max((global_step - 1) / warmup_steps, 0.0), 1.0)
    return float(maximum * progress)


__all__ = [
    "build_credit_residual",
    "compute_monte_carlo_credit",
    "compute_step_uncertainty",
    "credit_advantage_coefficient",
    "expand_step_credit_to_tokens",
    "merge_anchor_credit",
    "sample_anchor_steps",
]
