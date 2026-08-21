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
"""Same-prompt semantic retrieval for provider-supplied step values."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _relative_step_midpoints(
    step_start_mask: torch.Tensor,
    step_end_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Return response-relative midpoint coordinates for each step."""

    starts = step_start_mask.detach().cpu().bool()
    ends = step_end_mask.detach().cpu().bool()
    active = response_mask.detach().cpu().bool()
    midpoints: list[torch.Tensor] = []
    for row in range(ends.shape[0]):
        active_positions = torch.nonzero(active[row], as_tuple=False).flatten()
        start_positions = torch.nonzero(starts[row], as_tuple=False).flatten()
        end_positions = torch.nonzero(ends[row], as_tuple=False).flatten()
        if start_positions.numel() != end_positions.numel():
            raise ValueError(f"Response {row}: step start/end counts differ")
        if end_positions.numel() == 0:
            raise ValueError(f"Response {row}: no step endpoint")

        position_to_rank = {int(position): rank for rank, position in enumerate(active_positions.tolist())}
        start_ranks = torch.tensor(
            [position_to_rank[int(position)] for position in start_positions],
            dtype=torch.float32,
        )
        end_ranks = torch.tensor(
            [position_to_rank[int(position)] for position in end_positions],
            dtype=torch.float32,
        )
        # The shared splitter excludes trailing EOS from its final endpoint.
        text_token_count = float(int(end_ranks[-1].item()) + 1)
        midpoints.append((start_ranks + end_ranks) / (2.0 * text_token_count))
    return midpoints


def _prompt_key(prompt_id: Any) -> Any:
    try:
        hash(prompt_id)
        return prompt_id
    except TypeError:
        return repr(prompt_id)


def validate_similarity_options(*, top_k: int, tau: float, position_window: float, iterations: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"similarity_top_k must be a positive integer, got {top_k!r}")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"similarity_tau must be finite and positive, got {tau!r}")
    if not math.isfinite(position_window) or not 0.0 <= position_window <= 1.0:
        raise ValueError(f"similarity_position_window must be finite and in [0, 1], got {position_window!r}")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError(f"similarity_iterations must be a positive integer, got {iterations!r}")


def compute_similarity_step_values(
    *,
    packed_step_embeddings: torch.Tensor,
    step_start_mask: torch.Tensor,
    step_end_mask: torch.Tensor,
    response_mask: torch.Tensor,
    outcomes: torch.Tensor,
    prompt_ids: np.ndarray,
    top_k: int,
    tau: float,
    position_window: float,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Estimate nonterminal values from other responses' semantically similar steps.

    Iteration zero associates every nonterminal source step with its response's
    final raw reward. Each synchronous propagation round then replaces that
    value with the temperature-weighted KNN estimate from the preceding round.
    Terminal values always remain the exact final rewards.
    """

    validate_similarity_options(
        top_k=top_k,
        tau=tau,
        position_window=position_window,
        iterations=iterations,
    )
    if packed_step_embeddings.ndim != 3:
        raise ValueError("packed_step_embeddings must have shape [batch, max_steps, hidden]")
    if step_start_mask.ndim != 2 or step_end_mask.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("step masks and response_mask must be rank-2")
    if step_start_mask.shape != step_end_mask.shape or response_mask.shape != step_end_mask.shape:
        raise ValueError("step_start_mask, step_end_mask, and response_mask must have identical shapes")

    batch_size = step_end_mask.shape[0]
    if packed_step_embeddings.shape[0] != batch_size:
        raise ValueError("Embedding and step-mask batch sizes differ")
    rewards = outcomes.detach().float().reshape(-1).cpu()
    if rewards.numel() != batch_size or not torch.isfinite(rewards).all():
        raise ValueError("outcomes must contain one finite value per response")
    prompt_ids = np.asarray(prompt_ids, dtype=object).reshape(-1)
    if len(prompt_ids) != batch_size:
        raise ValueError("prompt_ids must contain one id per response")

    starts = step_start_mask.detach().cpu().bool()
    ends = step_end_mask.detach().cpu().bool()
    active = response_mask.detach().cpu().bool()
    if torch.any(starts & ~active) or torch.any(ends & ~active):
        raise ValueError("Step boundaries may select only active response tokens")
    step_counts = ends.sum(dim=-1, dtype=torch.long)
    if torch.any(step_counts <= 0):
        raise ValueError("Every response must contain at least one step")
    if not torch.equal(starts.sum(dim=-1), step_counts):
        raise ValueError("Every step endpoint must have exactly one step start")
    if int(step_counts.max().item()) > packed_step_embeddings.shape[1]:
        raise ValueError("Packed embedding width is smaller than a response's step count")

    embeddings = F.normalize(packed_step_embeddings.detach().float().cpu(), p=2, dim=-1, eps=1e-12)
    midpoints = _relative_step_midpoints(starts, ends, active)
    rows_by_prompt: dict[Any, list[int]] = defaultdict(list)
    for row, prompt_id in enumerate(prompt_ids.tolist()):
        rows_by_prompt[_prompt_key(prompt_id)].append(row)

    # values[row][step] is the source value consumed by the next synchronous
    # propagation round. Initially every step carries its response outcome.
    values = [rewards[row].repeat(int(step_counts[row].item())) for row in range(batch_size)]
    retrieved_targets = 0
    fallback_targets = 0
    selected_neighbors = 0
    eligible_neighbors = 0
    selected_cosine_sum = 0.0
    effective_neighbor_sum = 0.0

    for _ in range(iterations):
        next_values = [row_values.clone() for row_values in values]
        for rows in rows_by_prompt.values():
            has_nonterminal = any(int(step_counts[row].item()) > 1 for row in rows)
            if len(rows) < 2 and has_nonterminal:
                raise ValueError(
                    "Similarity step-value retrieval needs at least two responses "
                    "for each prompt with nonterminal steps"
                )
            for target_row in rows:
                target_count = int(step_counts[target_row].item())
                for target_step in range(target_count - 1):
                    candidate_rows: list[int] = []
                    candidate_steps: list[int] = []
                    for source_row in rows:
                        if source_row == target_row:
                            continue
                        source_count = int(step_counts[source_row].item())
                        for source_step in range(source_count - 1):
                            if (
                                abs(float(midpoints[target_row][target_step] - midpoints[source_row][source_step]))
                                <= position_window
                            ):
                                candidate_rows.append(source_row)
                                candidate_steps.append(source_step)

                    if not candidate_rows:
                        other_rows = [row for row in rows if row != target_row]
                        if not other_rows:
                            raise ValueError("Cannot form a leave-one-response-out similarity fallback")
                        next_values[target_row][target_step] = rewards[other_rows].mean()
                        fallback_targets += 1
                        continue

                    candidate_vectors = torch.stack(
                        [embeddings[row, step] for row, step in zip(candidate_rows, candidate_steps, strict=True)]
                    )
                    similarities = candidate_vectors @ embeddings[target_row, target_step]
                    neighbor_count = min(top_k, similarities.numel())
                    selected_similarity, selected_indices = torch.topk(similarities, k=neighbor_count)
                    weights = torch.softmax(selected_similarity / tau, dim=0)
                    neighbor_values = torch.stack(
                        [
                            values[candidate_rows[int(index)]][candidate_steps[int(index)]]
                            for index in selected_indices.tolist()
                        ]
                    )
                    next_values[target_row][target_step] = torch.sum(weights * neighbor_values)

                    retrieved_targets += 1
                    eligible_neighbors += len(candidate_rows)
                    selected_neighbors += neighbor_count
                    selected_cosine_sum += float(selected_similarity.sum().item())
                    effective_neighbor_sum += float((1.0 / weights.square().sum()).item())
        values = next_values

    dense_values = torch.zeros_like(ends, dtype=torch.float32)
    for row in range(batch_size):
        end_positions = torch.nonzero(ends[row], as_tuple=False).flatten()
        row_values = values[row].clone()
        row_values[-1] = rewards[row]
        dense_values[row, end_positions] = row_values

    total_targets = retrieved_targets + fallback_targets
    metrics = {
        "similarity_step_value/retrieval_coverage": (
            float(retrieved_targets / total_targets) if total_targets else 1.0
        ),
        "similarity_step_value/fallback_fraction": (float(fallback_targets / total_targets) if total_targets else 0.0),
        "similarity_step_value/eligible_neighbors_mean": (
            float(eligible_neighbors / retrieved_targets) if retrieved_targets else 0.0
        ),
        "similarity_step_value/selected_neighbors_mean": (
            float(selected_neighbors / retrieved_targets) if retrieved_targets else 0.0
        ),
        "similarity_step_value/selected_cosine_mean": (
            float(selected_cosine_sum / selected_neighbors) if selected_neighbors else 0.0
        ),
        "similarity_step_value/effective_neighbors_mean": (
            float(effective_neighbor_sum / retrieved_targets) if retrieved_targets else 0.0
        ),
    }
    return dense_values.to(device=step_end_mask.device), metrics


def compute_similarity_step_value_diagnostics(
    *,
    step_values: torch.Tensor,
    step_end_mask: torch.Tensor,
    final_rewards: torch.Tensor,
) -> dict[str, float]:
    """Summarize raw-reward step values without probability-only metrics."""

    values = step_values.detach().float().cpu()
    ends = step_end_mask.detach().bool().cpu()
    rewards = final_rewards.detach().float().reshape(-1).cpu()
    if values.ndim != 2 or ends.shape != values.shape:
        raise ValueError("Similarity diagnostics require matching rank-2 value and endpoint tensors")
    if rewards.numel() != values.shape[0] or not torch.isfinite(rewards).all():
        raise ValueError("Similarity diagnostics require one finite reward per response")

    trajectory_values = []
    row_mses = []
    row_maes = []
    early_mses = []
    late_mses = []
    row_delta_abs = []
    row_delta_near_zero = []
    step_counts = ends.sum(dim=-1)
    for row in range(values.shape[0]):
        row_values = values[row, ends[row]]
        if row_values.numel() == 0 or not torch.isfinite(row_values).all():
            raise ValueError("Similarity diagnostics require finite values at every step endpoint")
        errors = row_values - rewards[row]
        trajectory_values.append(row_values.mean())
        row_mses.append(errors.square().mean())
        row_maes.append(errors.abs().mean())
        early_mses.append(errors[0].square())
        late_mses.append(errors[-1].square())
        if row_values.numel() > 1:
            deltas = row_values[1:] - row_values[:-1]
            row_delta_abs.append(deltas.abs().mean())
            row_delta_near_zero.append((deltas.abs() <= 1e-6).float().mean())

    trajectory_values_tensor = torch.stack(trajectory_values)
    return {
        "similarity_step_value/avg_steps_per_response": float(step_counts.float().mean().item()),
        "similarity_step_value/final_reward_mean": float(rewards.mean().item()),
        "similarity_step_value/final_reward_std": float(rewards.std(unbiased=False).item()),
        "similarity_step_value/value_mean": float(trajectory_values_tensor.mean().item()),
        "similarity_step_value/value_std": float(trajectory_values_tensor.std(unbiased=False).item()),
        "similarity_step_value/preupdate_mse": float(torch.stack(row_mses).mean().item()),
        "similarity_step_value/preupdate_mae": float(torch.stack(row_maes).mean().item()),
        "similarity_step_value/depth_q1_mse": float(torch.stack(early_mses).mean().item()),
        "similarity_step_value/depth_q4_mse": float(torch.stack(late_mses).mean().item()),
        "similarity_step_value/delta_abs_mean": (
            float(torch.stack(row_delta_abs).mean().item()) if row_delta_abs else 0.0
        ),
        "similarity_step_value/delta_near_zero_fraction": (
            float(torch.stack(row_delta_near_zero).mean().item()) if row_delta_near_zero else 0.0
        ),
    }


__all__ = [
    "compute_similarity_step_value_diagnostics",
    "compute_similarity_step_values",
    "validate_similarity_options",
]
