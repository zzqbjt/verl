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

"""Rank-preserving prompt-center calibration for Step Probe values.

The Probe itself and its BCE update stay unchanged.  This module only shifts
the pre-update endpoint logits used by the advantage estimator.  A prompt gets
one shared logit offset, so all of its Probe logit differences are preserved
before the verified terminal endpoint is restored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from verl.utils import as_torch_index

_AUDIT_STATE_FORMAT_VERSION = 1
_AUDIT_FIT_RIDGE = 1.0e-4
_AUDIT_FIT_MIN_CENTER_STD = 1.0e-6
_AUDIT_FIT_MAX_ABS_COEFFICIENT = 50.0


@dataclass(frozen=True)
class PromptCenterAuditFit:
    """Result of one group-binomial prompt-center fit."""

    slope: float
    intercept: float
    updated: bool
    reason: str
    iterations: int
    group_count: int


def _binomial_log_likelihood(
    coefficients: np.ndarray,
    design: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
    anchor: np.ndarray,
) -> float:
    logits = design @ coefficients
    # log(sigmoid(x)) and log(1-sigmoid(x)) without overflow.
    value = np.sum(successes * -np.logaddexp(0.0, -logits) + (totals - successes) * -np.logaddexp(0.0, logits))
    value -= 0.5 * _AUDIT_FIT_RIDGE * float(np.square(coefficients - anchor).sum())
    return float(value)


def fit_group_binomial_prompt_center_map(
    centers: Sequence[float] | np.ndarray,
    successes: Sequence[int] | np.ndarray,
    totals: Sequence[int] | np.ndarray,
    *,
    previous_slope: float,
    previous_intercept: float,
    max_iterations: int = 100,
    tolerance: float = 1.0e-9,
) -> PromptCenterAuditFit:
    """Fit ``sigmoid(slope * center + intercept)`` to grouped outcomes.

    The implementation uses a tiny, deterministic CPU IRLS solve.  The weak
    anchor ridge is a numerical stabilizer, not a user-facing hyperparameter.
    Complete positive-slope separation, degenerate outcomes, non-finite input,
    non-positive fitted slopes, and failed solves all retain the previous map.
    """

    old_slope = float(previous_slope)
    old_intercept = float(previous_intercept)
    group_count = int(np.asarray(centers).size)

    def unchanged(reason: str, iterations: int = 0) -> PromptCenterAuditFit:
        return PromptCenterAuditFit(
            slope=old_slope,
            intercept=old_intercept,
            updated=False,
            reason=reason,
            iterations=iterations,
            group_count=group_count,
        )

    if not np.isfinite(old_slope) or old_slope <= 0.0 or not np.isfinite(old_intercept):
        raise ValueError("Previous prompt-center parameters must be finite and slope must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    center_array = np.asarray(centers, dtype=np.float64).reshape(-1)
    success_array = np.asarray(successes, dtype=np.float64).reshape(-1)
    total_array = np.asarray(totals, dtype=np.float64).reshape(-1)
    if not (center_array.size == success_array.size == total_array.size):
        raise ValueError("centers, successes, and totals must have the same length")
    if center_array.size < 2:
        return unchanged("too_few_groups")
    if (
        not np.isfinite(center_array).all()
        or not np.isfinite(success_array).all()
        or not np.isfinite(total_array).all()
    ):
        return unchanged("nonfinite_input")
    if np.any(total_array <= 0.0) or np.any(success_array < 0.0) or np.any(success_array > total_array):
        raise ValueError("Each grouped outcome must satisfy 0 <= successes <= totals and totals > 0")
    if float(np.std(center_array)) <= _AUDIT_FIT_MIN_CENTER_STD:
        return unchanged("constant_center")

    total_successes = float(success_array.sum())
    total_trials = float(total_array.sum())
    if total_successes <= 0.0 or total_successes >= total_trials:
        return unchanged("single_outcome")

    mixed = (success_array > 0.0) & (success_array < total_array)
    pure_positive = center_array[success_array == total_array]
    pure_negative = center_array[success_array == 0.0]
    if (
        not mixed.any()
        and pure_positive.size > 0
        and pure_negative.size > 0
        and float(pure_negative.max()) <= float(pure_positive.min())
    ):
        return unchanged("positive_slope_separation")

    design = np.column_stack((center_array, np.ones_like(center_array)))
    anchor = np.asarray([old_slope, old_intercept], dtype=np.float64)
    coefficients = anchor.copy()
    objective = _binomial_log_likelihood(coefficients, design, success_array, total_array, anchor)
    if not np.isfinite(objective):
        return unchanged("nonfinite_objective")

    converged = False
    completed_iterations = 0
    for iteration in range(1, max_iterations + 1):
        completed_iterations = iteration
        logits = design @ coefficients
        probabilities = np.empty_like(logits)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1.0 + exp_logits)
        weights = total_array * probabilities * (1.0 - probabilities)
        information = design.T @ (weights[:, None] * design)
        information += _AUDIT_FIT_RIDGE * np.eye(2, dtype=np.float64)
        gradient = design.T @ (success_array - total_array * probabilities)
        gradient -= _AUDIT_FIT_RIDGE * (coefficients - anchor)
        if not np.isfinite(information).all() or not np.isfinite(gradient).all():
            return unchanged("nonfinite_newton_system", completed_iterations)
        try:
            direction = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return unchanged("singular_newton_system", completed_iterations)
        if not np.isfinite(direction).all():
            return unchanged("nonfinite_newton_step", completed_iterations)

        step_size = 1.0
        accepted = False
        candidate = coefficients
        candidate_objective = objective
        for _ in range(32):
            candidate = coefficients + step_size * direction
            if candidate[0] > 0.0 and np.max(np.abs(candidate)) <= _AUDIT_FIT_MAX_ABS_COEFFICIENT:
                candidate_objective = _binomial_log_likelihood(
                    candidate,
                    design,
                    success_array,
                    total_array,
                    anchor,
                )
                if np.isfinite(candidate_objective) and candidate_objective >= objective - 1.0e-12:
                    accepted = True
                    break
            step_size *= 0.5
        if not accepted:
            return unchanged("line_search_failed", completed_iterations)

        scaled_step = step_size * direction
        coefficients = candidate
        objective = candidate_objective
        if float(np.max(np.abs(scaled_step))) <= tolerance * (1.0 + float(np.max(np.abs(coefficients)))):
            converged = True
            break

    if not converged:
        return unchanged("not_converged", completed_iterations)
    if not np.isfinite(coefficients).all() or coefficients[0] <= 0.0:
        return unchanged("invalid_coefficients", completed_iterations)
    return PromptCenterAuditFit(
        slope=float(coefficients[0]),
        intercept=float(coefficients[1]),
        updated=True,
        reason="fit",
        iterations=completed_iterations,
        group_count=group_count,
    )


def aggregate_prompt_center_audit_groups(
    trajectory_logit_means: torch.Tensor,
    endpoint_counts: torch.Tensor,
    ready_next: torch.Tensor,
    prompt_ids: np.ndarray,
    targets: torch.Tensor,
    *,
    expected_group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-trajectory audit outputs into trajectory-balanced groups."""

    means = trajectory_logit_means.detach().cpu().float().reshape(-1)
    counts = endpoint_counts.detach().cpu().long().reshape(-1)
    ready = ready_next.detach().cpu().bool().reshape(-1)
    binary_targets = targets.detach().cpu().float().reshape(-1)
    ids = np.asarray(prompt_ids, dtype=object).reshape(-1)
    row_count = means.numel()
    if not (counts.numel() == ready.numel() == binary_targets.numel() == ids.size == row_count):
        raise ValueError("Audit trajectory fields must have identical lengths")
    if isinstance(expected_group_size, bool) or not isinstance(expected_group_size, int) or expected_group_size <= 0:
        raise ValueError("expected_group_size must be a positive integer")
    if row_count == 0:
        raise ValueError("Audit batch must contain at least one trajectory")
    if not ready.all():
        raise RuntimeError("Audit logits are not ready for the next update")
    if not torch.isfinite(means).all() or torch.any(counts <= 0):
        raise ValueError("Audit means must be finite and every trajectory must contain an endpoint")
    if not torch.isfinite(binary_targets).all() or torch.any(
        (binary_targets != 0.0) & (binary_targets != 1.0)
    ):
        raise ValueError("Audit targets must be finite binary outcomes")

    group_centers: list[float] = []
    group_successes: list[int] = []
    group_totals: list[int] = []
    for prompt_id in sorted(set(ids.tolist()), key=str):
        rows = np.flatnonzero(ids == prompt_id)
        if rows.size != expected_group_size:
            raise ValueError(
                f"Audit prompt {prompt_id!r} has {rows.size} trajectories; expected {expected_group_size}"
            )
        torch_rows = torch.as_tensor(rows, dtype=torch.long)
        group_centers.append(float(means.index_select(0, torch_rows).mean().item()))
        group_successes.append(int(binary_targets.index_select(0, torch_rows).sum().item()))
        group_totals.append(expected_group_size)
    return (
        np.asarray(group_centers, dtype=np.float64),
        np.asarray(group_successes, dtype=np.int64),
        np.asarray(group_totals, dtype=np.int64),
    )


class LaggedPromptCenterAuditState:
    """Driver-owned active/pending state for causally lagged audit calibration."""

    def __init__(
        self,
        *,
        initial_slope: float,
        initial_intercept: float,
        target_key: str,
        group_size: int,
        audit_groups: int,
        rolling_window: int,
        seed: int,
    ) -> None:
        if not np.isfinite(float(initial_slope)) or float(initial_slope) <= 0.0:
            raise ValueError("initial_slope must be finite and positive")
        if not np.isfinite(float(initial_intercept)):
            raise ValueError("initial_intercept must be finite")
        if not isinstance(target_key, str) or not target_key:
            raise ValueError("target_key must be a non-empty string")
        for name, value in (
            ("group_size", group_size),
            ("audit_groups", audit_groups),
            ("rolling_window", rolling_window),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        self.active_slope = float(initial_slope)
        self.active_intercept = float(initial_intercept)
        self.fingerprint = {
            "target_key": target_key,
            "group_size": int(group_size),
            "audit_groups": int(audit_groups),
            "rolling_window": int(rolling_window),
            "seed": int(seed),
        }
        self.entries: list[dict[str, Any]] = []
        self.last_audit_global_step = -1
        self.fit_count = 0
        self.skip_count = 0
        self._pending: tuple[float, float] | None = None

    @property
    def active_parameters(self) -> tuple[float, float]:
        return self.active_slope, self.active_intercept

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def stage(
        self,
        *,
        centers: Sequence[float] | np.ndarray,
        successes: Sequence[int] | np.ndarray,
        totals: Sequence[int] | np.ndarray,
        global_step: int,
    ) -> PromptCenterAuditFit:
        """Append one audit update and stage, but do not activate, its fit."""

        if self._pending is not None:
            raise RuntimeError("Cannot stage a second audit fit before committing the pending parameters")
        if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        if global_step <= self.last_audit_global_step:
            raise RuntimeError(
                f"Audit global_step must increase strictly: {global_step} <= {self.last_audit_global_step}"
            )

        center_array = np.asarray(centers, dtype=np.float64).reshape(-1)
        success_array = np.asarray(successes, dtype=np.int64).reshape(-1)
        total_array = np.asarray(totals, dtype=np.int64).reshape(-1)
        expected_groups = int(self.fingerprint["audit_groups"])
        expected_group_size = int(self.fingerprint["group_size"])
        if not (center_array.size == success_array.size == total_array.size == expected_groups):
            raise ValueError(f"Each audit update must contain exactly {expected_groups} complete prompt groups")
        if np.any(total_array != expected_group_size):
            raise ValueError(f"Every audit group must contain exactly {expected_group_size} trajectories")
        if not np.isfinite(center_array).all():
            raise ValueError("Audit centers must be finite")
        if np.any(success_array < 0) or np.any(success_array > total_array):
            raise ValueError("Audit success counts must lie between zero and the group total")

        self.entries.append(
            {
                "global_step": int(global_step),
                "centers": center_array.tolist(),
                "successes": success_array.tolist(),
                "totals": total_array.tolist(),
            }
        )
        window = int(self.fingerprint["rolling_window"])
        self.entries = self.entries[-window:]
        self.last_audit_global_step = int(global_step)

        fit = fit_group_binomial_prompt_center_map(
            np.concatenate([np.asarray(entry["centers"], dtype=np.float64) for entry in self.entries]),
            np.concatenate([np.asarray(entry["successes"], dtype=np.int64) for entry in self.entries]),
            np.concatenate([np.asarray(entry["totals"], dtype=np.int64) for entry in self.entries]),
            previous_slope=self.active_slope,
            previous_intercept=self.active_intercept,
        )
        self._pending = (fit.slope, fit.intercept)
        if fit.updated:
            self.fit_count += 1
        else:
            self.skip_count += 1
        return fit

    def stage_unchanged(self, *, global_step: int, reason: str) -> PromptCenterAuditFit:
        """Stage an unchanged map when the actor cannot produce a valid audit."""

        if self._pending is not None:
            raise RuntimeError("Cannot stage a second audit decision before committing")
        if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        if global_step <= self.last_audit_global_step:
            raise RuntimeError(
                f"Audit global_step must increase strictly: {global_step} <= {self.last_audit_global_step}"
            )
        self.last_audit_global_step = int(global_step)
        self._pending = self.active_parameters
        self.skip_count += 1
        return PromptCenterAuditFit(
            slope=self.active_slope,
            intercept=self.active_intercept,
            updated=False,
            reason=str(reason),
            iterations=0,
            group_count=0,
        )

    def commit(self) -> tuple[float, float]:
        """Activate the staged fit for the next training update."""

        if self._pending is None:
            raise RuntimeError("No pending audit parameters to commit")
        self.active_slope, self.active_intercept = self._pending
        self._pending = None
        return self.active_parameters

    def state_dict(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("Prompt-center audit state may only be checkpointed at an update boundary")
        return {
            "format_version": _AUDIT_STATE_FORMAT_VERSION,
            "fingerprint": dict(self.fingerprint),
            "active_slope": self.active_slope,
            "active_intercept": self.active_intercept,
            "entries": [
                {
                    "global_step": int(entry["global_step"]),
                    "centers": list(entry["centers"]),
                    "successes": list(entry["successes"]),
                    "totals": list(entry["totals"]),
                }
                for entry in self.entries
            ],
            "last_audit_global_step": self.last_audit_global_step,
            "fit_count": self.fit_count,
            "skip_count": self.skip_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("Prompt-center audit checkpoint must be a mapping")
        format_version = state.get("format_version", None)
        if type(format_version) is not int or format_version != _AUDIT_STATE_FORMAT_VERSION:
            raise ValueError("Unsupported prompt-center audit checkpoint format")
        loaded_fingerprint = state.get("fingerprint", None)
        if not isinstance(loaded_fingerprint, Mapping) or set(loaded_fingerprint) != set(self.fingerprint):
            raise ValueError(
                "Prompt-center audit checkpoint fingerprint does not match the active target/group/window/seed config"
            )
        if any(
            type(loaded_fingerprint[key]) is not type(expected) or loaded_fingerprint[key] != expected
            for key, expected in self.fingerprint.items()
        ):
            raise ValueError(
                "Prompt-center audit checkpoint fingerprint does not match the active target/group/window/seed config"
            )
        slope = state.get("active_slope", None)
        intercept = state.get("active_intercept", None)
        if type(slope) is not float or type(intercept) is not float:
            raise ValueError("Prompt-center audit checkpoint active parameters must be floats")
        if not np.isfinite(slope) or slope <= 0.0 or not np.isfinite(intercept):
            raise ValueError("Prompt-center audit checkpoint contains invalid active parameters")
        entries = state.get("entries", None)
        if not isinstance(entries, list) or len(entries) > int(self.fingerprint["rolling_window"]):
            raise ValueError("Prompt-center audit checkpoint contains an invalid FIFO")

        validated_entries: list[dict[str, Any]] = []
        previous_step = -1
        expected_groups = int(self.fingerprint["audit_groups"])
        expected_total = int(self.fingerprint["group_size"])
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("Prompt-center audit FIFO entries must be mappings")
            step = entry.get("global_step", None)
            raw_centers = entry.get("centers", None)
            raw_successes = entry.get("successes", None)
            raw_totals = entry.get("totals", None)
            if type(step) is not int:
                raise ValueError("Prompt-center audit FIFO global steps must be integers")
            if not all(isinstance(values, list) for values in (raw_centers, raw_successes, raw_totals)):
                raise ValueError("Prompt-center audit FIFO arrays must be lists")
            if any(type(value) is not float for value in raw_centers):
                raise ValueError("Prompt-center audit FIFO centers must be floats")
            if any(type(value) is not int for values in (raw_successes, raw_totals) for value in values):
                raise ValueError("Prompt-center audit FIFO outcomes must be integers")
            centers = np.asarray(raw_centers, dtype=np.float64).reshape(-1)
            successes = np.asarray(raw_successes, dtype=np.int64).reshape(-1)
            totals = np.asarray(raw_totals, dtype=np.int64).reshape(-1)
            if step <= previous_step:
                raise ValueError("Prompt-center audit FIFO steps must increase strictly")
            if not (centers.size == successes.size == totals.size == expected_groups):
                raise ValueError("Prompt-center audit FIFO contains an incomplete update")
            if not np.isfinite(centers).all() or np.any(totals != expected_total):
                raise ValueError("Prompt-center audit FIFO contains invalid centers or group totals")
            if np.any(successes < 0) or np.any(successes > totals):
                raise ValueError("Prompt-center audit FIFO contains invalid success counts")
            validated_entries.append(
                {
                    "global_step": step,
                    "centers": centers.tolist(),
                    "successes": successes.tolist(),
                    "totals": totals.tolist(),
                }
            )
            previous_step = step

        last_step = state.get("last_audit_global_step", None)
        if type(last_step) is not int or last_step < -1:
            raise ValueError("Prompt-center audit checkpoint contains an invalid last audit step")
        if validated_entries and validated_entries[-1]["global_step"] > last_step:
            raise ValueError("Prompt-center audit FIFO extends past last_audit_global_step")
        fit_count = state.get("fit_count", None)
        skip_count = state.get("skip_count", None)
        if type(fit_count) is not int or type(skip_count) is not int or fit_count < 0 or skip_count < 0:
            raise ValueError("Prompt-center audit checkpoint contains invalid fit counters")

        self.active_slope = slope
        self.active_intercept = intercept
        self.entries = validated_entries
        self.last_audit_global_step = last_step
        self.fit_count = fit_count
        self.skip_count = skip_count
        self._pending = None


def prompt_center_logit_offsets(
    trajectory_endpoint_logit_means: torch.Tensor,
    prompt_ids: np.ndarray,
    *,
    slope: float,
    intercept: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one shared calibration offset per response and prompt.

    Each input scalar must be the mean of *all* pre-update endpoint logits in
    one trajectory.  Averaging those scalars within a prompt makes the prompt
    center trajectory-balanced rather than endpoint-count-balanced.
    """

    trajectory_means = trajectory_endpoint_logit_means.detach().reshape(-1)
    if not trajectory_means.is_floating_point():
        raise ValueError("trajectory_endpoint_logit_means must be floating point")
    if not torch.isfinite(trajectory_means).all():
        raise ValueError("trajectory_endpoint_logit_means must be finite")
    if len(prompt_ids) != trajectory_means.numel():
        raise ValueError("prompt_ids and trajectory_endpoint_logit_means must contain the same number of responses")
    if not np.isfinite(float(slope)) or float(slope) <= 0.0:
        raise ValueError(f"prompt-center calibration slope must be finite and positive, got {slope}")
    if not np.isfinite(float(intercept)):
        raise ValueError(f"prompt-center calibration intercept must be finite, got {intercept}")

    group_index = as_torch_index(prompt_ids, device=trajectory_means.device)
    response_offsets = torch.empty_like(trajectory_means)
    prompt_centers = []
    for group_id in torch.unique(group_index).tolist():
        rows = torch.nonzero(group_index == group_id, as_tuple=False).flatten()
        center = trajectory_means.index_select(0, rows).mean()
        prompt_centers.append(center)
        offset = (float(slope) - 1.0) * center + float(intercept)
        response_offsets.index_fill_(0, rows, offset)
    centers = torch.stack(prompt_centers) if prompt_centers else trajectory_means.new_empty((0,))
    return response_offsets, centers


@torch.no_grad()
def apply_rank_preserving_prompt_center_calibration(
    step_values: torch.Tensor,
    step_end_mask: torch.Tensor,
    trajectory_endpoint_logit_means: torch.Tensor,
    prompt_ids: np.ndarray,
    terminal_targets: torch.Tensor,
    *,
    enabled: bool,
    slope: float,
    intercept: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Shift every predicted endpoint logit by its prompt's shared offset.

    The final endpoint is overwritten with the verifier target after the
    transformation.  Consequently, the transformation preserves all
    within-prompt differences among predicted (nonterminal) logits while the
    advantage path retains its exact terminal anchor.
    """

    if not enabled:
        return step_values, {"enabled": False}

    if step_values.ndim != 2:
        raise ValueError(f"step_values must be rank 2, got shape {tuple(step_values.shape)}")
    if not step_values.is_floating_point():
        raise ValueError("step_values must be floating point")
    if step_end_mask.shape != step_values.shape:
        raise ValueError(f"step_end_mask must have shape {tuple(step_values.shape)}, got {tuple(step_end_mask.shape)}")
    end_mask = step_end_mask.bool()
    if torch.any(end_mask.sum(dim=-1) == 0):
        raise ValueError("Every response must contain at least one step endpoint")

    batch_size, response_length = step_values.shape
    targets = terminal_targets.detach().to(device=step_values.device, dtype=step_values.dtype).reshape(-1)
    if targets.numel() != batch_size:
        raise ValueError("terminal_targets must contain one value per response")
    if not torch.isfinite(targets).all() or torch.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("terminal_targets must be finite and lie in [0, 1]")

    offsets, centers = prompt_center_logit_offsets(
        trajectory_endpoint_logit_means.to(device=step_values.device),
        prompt_ids,
        slope=slope,
        intercept=intercept,
    )
    terminal_positions = (
        torch.arange(response_length, device=step_values.device)
        .expand_as(end_mask)
        .masked_fill(~end_mask, -1)
        .max(dim=-1)
        .values
    )
    terminal_mask = torch.zeros_like(end_mask)
    terminal_mask[torch.arange(batch_size, device=step_values.device), terminal_positions] = True
    predicted_endpoint_mask = end_mask & ~terminal_mask

    calibrated = step_values.clone()
    if predicted_endpoint_mask.any():
        probabilities = calibrated[predicted_endpoint_mask]
        if not torch.isfinite(probabilities).all() or torch.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("Predicted step values must be finite and lie in [0, 1]")
        endpoint_offsets = offsets.unsqueeze(-1).expand_as(step_values)[predicted_endpoint_mask]
        # Avoid a needless logit/sigmoid round trip for the safe identity
        # default.  This keeps every nonterminal value bitwise unchanged.
        if torch.any(endpoint_offsets != 0.0):
            # The probabilities came from a float32 sigmoid.  Clamp only values
            # rounded to an exact boundary before reconstructing their logits.
            epsilon = torch.finfo(probabilities.dtype).eps
            logits = torch.logit(probabilities, eps=epsilon)
            calibrated[predicted_endpoint_mask] = torch.sigmoid(logits + endpoint_offsets)

    calibrated[
        torch.arange(batch_size, device=step_values.device),
        terminal_positions,
    ] = targets

    offsets_float = offsets.float()
    centers_float = centers.float()
    metrics = {
        "enabled": True,
        "prompt_count": int(centers.numel()),
        "predicted_endpoint_count": int(predicted_endpoint_mask.sum().item()),
        "center_mean": float(centers_float.mean().item()) if centers.numel() else 0.0,
        "center_std": float(centers_float.std(unbiased=False).item()) if centers.numel() else 0.0,
        "offset_mean": float(offsets_float.mean().item()) if offsets.numel() else 0.0,
        "offset_std": float(offsets_float.std(unbiased=False).item()) if offsets.numel() else 0.0,
        "offset_min": float(offsets_float.min().item()) if offsets.numel() else 0.0,
        "offset_max": float(offsets_float.max().item()) if offsets.numel() else 0.0,
    }
    return calibrated, metrics


__all__ = [
    "LaggedPromptCenterAuditState",
    "PromptCenterAuditFit",
    "aggregate_prompt_center_audit_groups",
    "apply_rank_preserving_prompt_center_calibration",
    "fit_group_binomial_prompt_center_map",
    "prompt_center_logit_offsets",
]
