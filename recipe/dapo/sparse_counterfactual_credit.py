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
"""Sparse counterfactual step-credit rollouts for the DAPO recipe."""

from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.sparse_counterfactual_credit import (
    compute_step_uncertainty,
    sample_anchor_steps,
)
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import auto_await


@dataclass(frozen=True)
class _SuffixRequest:
    anchor_index: int
    source_index: int
    kind: str
    response_prefix: tuple[int, ...]


@dataclass(frozen=True)
class _ScoredSuffixes:
    correctness: torch.Tensor
    rewards: torch.Tensor


@dataclass(frozen=True)
class _BranchCandidate:
    prompt_uid: object
    anchor_index: int
    source_index: int
    kind: str
    response_prefix: tuple[int, ...]
    anchor_entropy: float
    correctness_variance: float
    request_indices: tuple[int, ...]


@dataclass(frozen=True)
class CounterfactualCreditRolloutResult:
    metrics: dict[str, float]
    training_branches: DataProto | None = None


class SparseCounterfactualCreditSupervisor:
    """Select sparse step anchors, sample exact suffixes, and score Q/V."""

    def __init__(self, tokenizer, trainer_config, rollout_manager, reward_loop_manager):
        self.tokenizer = tokenizer
        self.trainer_config = trainer_config
        self.config = trainer_config.algorithm.sparse_counterfactual_credit
        self.rollout_config = trainer_config.actor_rollout_ref.rollout
        self.reward_loop_manager = reward_loop_manager
        servers = list(zip(rollout_manager.server_addresses, rollout_manager.server_handles, strict=True))
        self.direct_server_manager = AsyncLLMServerManager(
            config=trainer_config,
            servers=servers,
            load_balancer_handle=rollout_manager.global_load_balancer,
        )

    @staticmethod
    def _response_prefix_ids(batch: DataProto, row: int, stop: int) -> list[int]:
        mask = batch.batch["response_mask"][row, :stop].bool()
        return [int(token_id) for token_id in batch.batch["responses"][row, :stop][mask].tolist()]

    @staticmethod
    def _direct_prompt_ids(batch: DataProto, row: int, response_prefix: list[int]) -> list[int]:
        prompt_width = batch.batch["prompts"].shape[-1]
        prompt_mask = batch.batch["attention_mask"][row, :prompt_width].bool()
        prompt_ids = batch.batch["prompts"][row][prompt_mask].tolist()
        return [int(token_id) for token_id in prompt_ids] + response_prefix

    def _counterfactual_response_limit(self) -> int:
        reward_config = self.trainer_config.get("reward", self.trainer_config.get("reward_model", {}))
        buffer_config = reward_config.get("reward_kwargs", {}).get("overlong_buffer_cfg", {}) or {}
        buffer_length = int(buffer_config.get("len", 0))
        response_length = int(self.trainer_config.data.max_response_length)
        if not 0 <= buffer_length < response_length:
            raise ValueError("MC overlong buffer length must be in [0, max_response_length).")
        return response_length - buffer_length

    def _max_new_tokens(self, prompt_length: int, response_prefix_length: int) -> int:
        data_config = self.trainer_config.data
        max_response_length = int(data_config.max_response_length)
        if response_prefix_length > max_response_length:
            raise ValueError(
                "Counterfactual response prefix exceeds data.max_response_length; "
                f"response_prefix_length={response_prefix_length}, "
                f"max_response_length={max_response_length}."
            )

        response_budget = self._counterfactual_response_limit() - response_prefix_length
        max_tokens = max(response_budget, 0)
        configured_cap = self.config.max_new_tokens
        if configured_cap is not None:
            max_tokens = min(max_tokens, int(configured_cap))
        max_model_len = self.rollout_config.get("max_model_len", None)
        if max_model_len is None:
            max_model_len = int(data_config.max_prompt_length) + int(data_config.max_response_length)
        max_tokens = min(max_tokens, max(int(max_model_len) - prompt_length, 0))
        return max_tokens

    def _sampling_params(self, prompt_length: int, response_prefix_length: int) -> dict[str, Any]:
        return {
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
            "top_k": int(self.config.top_k),
            "repetition_penalty": float(self.config.repetition_penalty),
            "max_tokens": self._max_new_tokens(prompt_length, response_prefix_length),
            "logprobs": False,
        }

    @auto_await
    async def _generate(
        self,
        prompt_ids: list[list[int]],
        response_prefix_lengths: list[int],
    ) -> list[Any]:
        if len(prompt_ids) != len(response_prefix_lengths):
            raise ValueError("Counterfactual prompts and response-prefix lengths must have the same size.")

        async def generate_one(ids: list[int], response_prefix_length: int):
            sampling_params = self._sampling_params(len(ids), response_prefix_length)
            if sampling_params["max_tokens"] == 0:
                return None
            return await self.direct_server_manager.generate(
                request_id=f"sparse-counterfactual-credit-{uuid.uuid4().hex}",
                prompt_ids=ids,
                sampling_params=sampling_params,
            )

        return await asyncio.gather(
            *(
                generate_one(ids, response_prefix_length)
                for ids, response_prefix_length in zip(prompt_ids, response_prefix_lengths, strict=True)
            ),
            return_exceptions=True,
        )

    def _make_reward_batch(
        self,
        batch: DataProto,
        source_indices: list[int],
        response_ids: list[list[int]],
    ) -> DataProto:
        if not response_ids:
            raise ValueError("Counterfactual scoring received an empty response list.")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Counterfactual scoring requires a tokenizer pad_token_id or eos_token_id.")
        response_width = max(max(map(len, response_ids)), 1)
        responses = torch.full(
            (len(response_ids), response_width),
            int(pad_token_id),
            dtype=batch.batch["responses"].dtype,
        )
        response_mask = torch.zeros(
            (len(response_ids), response_width),
            dtype=batch.batch["attention_mask"].dtype,
        )
        for output_row, ids in enumerate(response_ids):
            if ids:
                responses[output_row, : len(ids)] = torch.tensor(ids, dtype=responses.dtype)
                response_mask[output_row, : len(ids)] = 1
        source_tensor = torch.tensor(source_indices, dtype=torch.long)
        prompts = batch.batch["prompts"].index_select(0, source_tensor)
        prompt_width = prompts.shape[-1]
        prompt_mask = batch.batch["attention_mask"].index_select(0, source_tensor)[:, :prompt_width]
        attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
        source_numpy = np.asarray(source_indices)
        non_tensors = {key: value[source_numpy] for key, value in batch.non_tensor_batch.items()}
        return DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat((prompts, responses), dim=-1),
                "attention_mask": attention_mask,
                "position_ids": compute_position_id_with_mask(attention_mask),
            },
            non_tensors=non_tensors,
        )

    def _score(
        self,
        batch: DataProto,
        source_indices: list[int],
        response_ids: list[list[int]],
    ) -> _ScoredSuffixes:
        reward_batch = self._make_reward_batch(batch, source_indices, response_ids)
        num_reward_workers = int(self.trainer_config.reward.num_workers)
        padded_batch, pad_size = pad_dataproto_to_divisor(reward_batch, num_reward_workers)
        reward_result = self.reward_loop_manager.compute_rm_score(padded_batch)
        reward_result = unpad_dataproto(reward_result, pad_size)
        correctness_key = str(self.config.correctness_key)
        if correctness_key not in reward_result.non_tensor_batch:
            raise KeyError(f"Reward result does not contain required raw correctness key {correctness_key!r}.")
        scores = torch.as_tensor(np.asarray(reward_result.non_tensor_batch[correctness_key], dtype=np.float32))
        if (
            scores.shape != (len(response_ids),)
            or torch.any(~torch.isfinite(scores))
            or torch.any((scores < 0) | (scores > 1))
        ):
            raise ValueError("Counterfactual verifier scores must be one finite scalar in [0, 1] per suffix.")
        rewards = reward_result.batch["rm_scores"].sum(dim=-1).detach().cpu().float()
        if rewards.shape != scores.shape or torch.any(~torch.isfinite(rewards)):
            raise ValueError("Counterfactual rewards must be one finite scalar per suffix.")
        return _ScoredSuffixes(correctness=scores, rewards=rewards)

    def _budget_anchor_candidates(
        self,
        step_end_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Keep endpoints within the MC budget, reserving a token for nonterminal Q."""

        if step_end_mask.ndim != 2 or response_mask.ndim != 2:
            raise ValueError("step_end_mask and response_mask must be rank-2")
        if step_end_mask.shape != response_mask.shape:
            raise ValueError("step_end_mask and response_mask must have the same shape")
        endpoints = step_end_mask.bool() & response_mask.bool()
        prefix_lengths = response_mask.long().cumsum(dim=-1)
        limit = self._counterfactual_response_limit()
        terminal = endpoints & (endpoints.long().cumsum(dim=-1) == endpoints.sum(dim=-1, keepdim=True))
        return endpoints & ((prefix_lengths < limit) | (terminal & (prefix_lengths == limit)))

    def _mc_branch_anchor_candidates(self, step_end_mask: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Additionally exclude single-step responses when training MC branches."""
        candidates = self._budget_anchor_candidates(step_end_mask, response_mask)
        multi_step_response = step_end_mask.bool().sum(dim=-1) > 1
        return candidates & multi_step_response.unsqueeze(-1)

    def mc_branch_eligible_row_mask(self, batch: DataProto) -> torch.Tensor:
        """Return rows containing at least one valid MC branch anchor."""

        if "step_end_mask" not in batch.batch or "uid" not in batch.non_tensor_batch:
            raise ValueError("MC branch prompt filtering requires step_end_mask and uid.")
        if "response_mask" not in batch.batch:
            raise ValueError("MC branch prompt filtering requires response_mask.")
        candidate_mask = self._mc_branch_anchor_candidates(
            batch.batch["step_end_mask"],
            batch.batch["response_mask"],
        )
        return candidate_mask.any(dim=-1)

    def mc_branch_eligible_prompt_uids(self, batch: DataProto) -> set[object]:
        """Return prompts with enough distinct multi-step responses for all anchors."""

        eligible_rows = self.mc_branch_eligible_row_mask(batch).tolist()
        eligible_counts: dict[object, int] = {}
        for uid, is_eligible in zip(batch.non_tensor_batch["uid"].tolist(), eligible_rows, strict=True):
            eligible_counts.setdefault(uid, 0)
            eligible_counts[uid] += int(is_eligible)
        required = int(self.config.anchors_per_group)
        return {uid for uid, count in eligible_counts.items() if count >= required}

    @staticmethod
    def _select_training_branch_groups(
        candidates: list[_BranchCandidate],
        prompt_uids: list[object],
        *,
        groups_per_prompt: int,
        seed: int,
        global_step: int,
    ) -> list[_BranchCandidate]:
        """Select groups by variance, anchor diversity, entropy, then seeded randomness."""

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(global_step) * 1_000_003)
        selected: list[_BranchCandidate] = []
        by_prompt: dict[object, list[_BranchCandidate]] = {}
        for candidate in candidates:
            by_prompt.setdefault(candidate.prompt_uid, []).append(candidate)

        for prompt_uid in dict.fromkeys(prompt_uids):
            available = list(by_prompt.get(prompt_uid, ()))
            if len(available) < groups_per_prompt:
                raise ValueError(
                    f"Prompt {prompt_uid!r} has only {len(available)} trainable MC prefix groups; "
                    f"branch_groups_per_prompt={groups_per_prompt}."
                )
            used_anchors: set[int] = set()
            for _ in range(groups_per_prompt):
                best_variance = max(candidate.correctness_variance for candidate in available)
                tied = [
                    candidate
                    for candidate in available
                    if math.isclose(candidate.correctness_variance, best_variance, rel_tol=0.0, abs_tol=1e-12)
                ]
                distinct = [candidate for candidate in tied if candidate.anchor_index not in used_anchors]
                if distinct:
                    tied = distinct
                best_entropy = max(candidate.anchor_entropy for candidate in tied)
                tied = [
                    candidate
                    for candidate in tied
                    if math.isclose(candidate.anchor_entropy, best_entropy, rel_tol=0.0, abs_tol=1e-12)
                ]
                choice = tied[int(torch.randint(len(tied), (1,), generator=generator).item())]
                selected.append(choice)
                used_anchors.add(choice.anchor_index)
                available.remove(choice)
        return selected

    def _make_training_branch_batch(
        self,
        batch: DataProto,
        requests: list[_SuffixRequest],
        full_responses: list[list[int]],
        continuations: list[list[int]],
        correctness_scores: torch.Tensor,
        reward_scores: torch.Tensor,
        q_estimates: torch.Tensor,
        v_estimates: torch.Tensor,
        selected_groups: list[_BranchCandidate],
        *,
        global_step: int,
    ) -> DataProto:
        selected_request_indices = [index for group in selected_groups for index in group.request_indices]
        source_indices = [requests[index].source_index for index in selected_request_indices]
        branches = batch.select_idxs(source_indices)
        response_width = batch.batch["responses"].shape[-1]
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("MC branch training requires a tokenizer pad_token_id or eos_token_id.")

        responses = torch.full_like(branches.batch["responses"], int(pad_token_id))
        full_response_mask = torch.zeros_like(branches.batch["response_mask"])
        train_response_mask = torch.zeros_like(branches.batch["response_mask"])
        prefix_lengths = torch.zeros(len(source_indices), dtype=torch.long)
        token_level_scores = torch.zeros_like(branches.batch["token_level_scores"], dtype=torch.float32)
        start_value_targets = torch.zeros(len(source_indices), dtype=torch.float32)
        group_uids: list[str] = []
        request_to_group: dict[int, _BranchCandidate] = {
            request_index: group for group in selected_groups for request_index in group.request_indices
        }
        for branch_row, request_index in enumerate(selected_request_indices):
            request = requests[request_index]
            response = full_responses[request_index]
            continuation = continuations[request_index]
            prefix_length = len(request.response_prefix)
            if not continuation:
                raise ValueError("A selected MC training branch contains no newly generated suffix tokens.")
            if len(response) > response_width:
                raise ValueError(f"MC training branch length {len(response)} exceeds response width {response_width}.")
            responses[branch_row, : len(response)] = torch.tensor(response, dtype=responses.dtype)
            full_response_mask[branch_row, : len(response)] = 1
            train_response_mask[branch_row, prefix_length : len(response)] = 1
            prefix_lengths[branch_row] = prefix_length
            token_level_scores[branch_row, len(response) - 1] = reward_scores[request_index]
            group = request_to_group[request_index]
            start_value_targets[branch_row] = (
                q_estimates[group.anchor_index] if group.kind == "q" else v_estimates[group.anchor_index]
            )
            group_uids.append(f"{group.prompt_uid}:mc-branch:{global_step}:{group.anchor_index}:{group.kind}")

        prompt_width = branches.batch["prompts"].shape[-1]
        prompt_mask = branches.batch["attention_mask"][:, :prompt_width]
        attention_mask = torch.cat((prompt_mask, full_response_mask), dim=-1)
        branches.batch["responses"] = responses
        branches.batch["response_mask"] = train_response_mask
        branches.batch["input_ids"] = torch.cat((branches.batch["prompts"], responses), dim=-1)
        branches.batch["attention_mask"] = attention_mask
        branches.batch["position_ids"] = compute_position_id_with_mask(attention_mask)
        branches.batch["token_level_scores"] = token_level_scores
        branches.batch["token_level_rewards"] = token_level_scores.clone()
        if "rm_scores" in branches.batch:
            branches.batch["rm_scores"] = token_level_scores.clone()
        branches.batch["credit_prefix_lengths"] = prefix_lengths
        branches.batch["is_mc_branch"] = torch.ones(len(branches), dtype=torch.bool)

        stale_keys = (
            "step_end_mask",
            "old_log_probs",
            "entropys",
            "ref_log_prob",
            "rollout_log_probs",
            "rollout_is_weights",
            "routed_experts",
            "credit_anchor_mask",
            "credit_anchor_targets",
            "credit_anchor_q_targets",
            "credit_anchor_v_targets",
            "credit_start_value_mask",
            "credit_start_value_train_mask",
            "credit_start_value_targets",
            "credit_terminal_value_mask",
            "credit_terminal_value_targets",
            "credit_predictions",
            "advantages",
            "returns",
            "counterfactual_credit_residual",
        )
        for key in stale_keys:
            if key in branches.batch:
                branches.batch.pop(key)
        branches.batch["credit_start_value_mask"] = torch.ones(len(branches), dtype=torch.bool)
        branches.batch["credit_start_value_train_mask"] = torch.zeros(len(branches), dtype=torch.bool)
        branches.batch["credit_start_value_targets"] = start_value_targets
        branches.batch["credit_terminal_value_mask"] = torch.ones(len(branches), dtype=torch.bool)
        branches.batch["credit_terminal_value_targets"] = correctness_scores[selected_request_indices].float()
        branches.non_tensor_batch["uid"] = np.asarray(group_uids, dtype=object)
        branches.non_tensor_batch[str(self.config.correctness_key)] = correctness_scores[
            selected_request_indices
        ].numpy()
        return branches

    def collect_targets(self, batch: DataProto, *, global_step: int) -> CounterfactualCreditRolloutResult:
        """Attach sparse anchor targets and optionally return trainable MC branches."""

        required = ("response_mask", "step_end_mask", "entropys")
        missing = [key for key in required if key not in batch.batch]
        if missing:
            raise ValueError("Counterfactual target collection is missing: " + ", ".join(missing))
        if "uid" not in batch.non_tensor_batch:
            raise ValueError("Counterfactual target collection requires uid groups.")
        correctness_key = str(self.config.correctness_key)
        if correctness_key not in batch.non_tensor_batch:
            raise KeyError(f"Original rollout batch lacks raw correctness key {correctness_key!r}.")

        step_uncertainty = compute_step_uncertainty(
            batch.batch["entropys"],
            batch.batch["step_end_mask"],
            batch.batch["response_mask"],
            top_ratio=float(self.config.entropy_top_ratio),
        )
        anchor_candidate_mask = batch.batch["step_end_mask"]
        if bool(self.config.train_mc_branches):
            anchor_candidate_mask = self._mc_branch_anchor_candidates(
                anchor_candidate_mask,
                batch.batch["response_mask"],
            )
        else:
            anchor_candidate_mask = self._budget_anchor_candidates(
                anchor_candidate_mask, batch.batch["response_mask"]
            )
        anchor_mask = sample_anchor_steps(
            step_uncertainty,
            anchor_candidate_mask,
            batch.non_tensor_batch["uid"].tolist(),
            temperature=float(self.config.sampling_temperature),
            uniform_mix=float(self.config.uniform_mix),
            anchors_per_group=int(self.config.anchors_per_group),
            seed=int(self.config.selection_seed),
            global_step=global_step,
        )
        anchor_coordinates = torch.nonzero(anchor_mask, as_tuple=False)
        anchor_entropies = step_uncertainty[anchor_mask].detach().cpu().float()
        original_scores = torch.as_tensor(np.asarray(batch.non_tensor_batch[correctness_key], dtype=np.float32))
        if torch.any(~torch.isfinite(original_scores)) or torch.any((original_scores < 0) | (original_scores > 1)):
            raise ValueError("Original correctness scores must be finite and in [0, 1].")

        # The state before the first step is shared by every response in a UID
        # group, so its value target is the group's empirical correctness.
        # Only one copy per UID is used to train the probe, while every response
        # carries the value for replacing predictions at this known boundary.
        group_rows: dict[object, list[int]] = {}
        uids = batch.non_tensor_batch["uid"].tolist()
        for row, uid in enumerate(uids):
            group_rows.setdefault(uid, []).append(row)
        initial_state_values = torch.empty_like(original_scores)
        initial_state_train_mask = torch.zeros(len(batch), dtype=torch.bool)
        for rows in group_rows.values():
            row_scores = original_scores[rows]
            initial_state_values[rows] = row_scores.mean()
            initial_state_train_mask[rows[0]] = True

        requests: list[_SuffixRequest] = []
        prompt_ids: list[list[int]] = []
        q_reward_sums = torch.zeros(anchor_coordinates.shape[0], dtype=torch.float32)
        q_reward_counts = torch.zeros(anchor_coordinates.shape[0], dtype=torch.long)
        v_reward_sums = torch.zeros(anchor_coordinates.shape[0], dtype=torch.float32)
        v_reward_counts = torch.zeros(anchor_coordinates.shape[0], dtype=torch.long)
        for anchor_index, coordinate in enumerate(anchor_coordinates.tolist()):
            row, endpoint = coordinate
            active_positions = (
                torch.nonzero(
                    batch.batch["response_mask"][row].bool(),
                    as_tuple=False,
                )
                .flatten()
                .tolist()
            )
            endpoints = (
                torch.nonzero(
                    batch.batch["step_end_mask"][row].bool(),
                    as_tuple=False,
                )
                .flatten()
                .tolist()
            )
            step_index = endpoints.index(endpoint)
            if step_index == 0:
                step_start = active_positions[0]
            else:
                previous_endpoint_rank = active_positions.index(endpoints[step_index - 1])
                step_start = active_positions[previous_endpoint_rank + 1]
            before_prefix = self._response_prefix_ids(batch, row, step_start)
            retained_prefix = self._response_prefix_ids(batch, row, endpoint + 1)
            is_initial_step = step_index == 0
            is_terminal_step = step_index == len(endpoints) - 1

            # At a terminal step Q(s, a) is exactly the observed final reward;
            # there is no post-step continuation to sample.
            if is_terminal_step:
                q_reward_sums[anchor_index] = original_scores[row]
                q_reward_counts[anchor_index] = 1
            else:
                for _ in range(int(self.config.num_samples)):
                    requests.append(_SuffixRequest(anchor_index, row, "q", tuple(retained_prefix)))
                    prompt_ids.append(self._direct_prompt_ids(batch, row, retained_prefix))

            # Before the first step, the whole original response group already
            # supplies an empirical value for the shared initial state.
            if is_initial_step:
                v_reward_sums[anchor_index] = initial_state_values[row]
                v_reward_counts[anchor_index] = 1
            else:
                for _ in range(int(self.config.num_samples)):
                    requests.append(_SuffixRequest(anchor_index, row, "v", tuple(before_prefix)))
                    prompt_ids.append(self._direct_prompt_ids(batch, row, before_prefix))

        outputs = self._generate(prompt_ids, [len(request.response_prefix) for request in requests]) if requests else []
        failures = [output for output in outputs if isinstance(output, BaseException)]
        if failures:
            raise RuntimeError(
                f"Counterfactual suffix rollout failed for {len(failures)}/{len(outputs)} requests: {failures[0]}"
            )
        full_responses = []
        continuations = []
        for request, output in zip(requests, outputs, strict=True):
            continuation = [] if output is None else [int(token_id) for token_id in output.token_ids]
            if output is not None and not continuation:
                eos_token_id = self.tokenizer.eos_token_id
                if eos_token_id is None:
                    raise RuntimeError("An empty suffix rollout requires tokenizer.eos_token_id for scoring.")
                continuation = [int(eos_token_id)]
            continuations.append(continuation)
            full_responses.append([*request.response_prefix, *continuation])
        if requests:
            scored_suffixes = self._score(
                batch,
                [request.source_index for request in requests],
                full_responses,
            )
            # Preserve compatibility with small test doubles and external
            # supervisors that returned only correctness before branch training
            # was introduced.
            if isinstance(scored_suffixes, torch.Tensor):
                sampled_scores = scored_suffixes
                sampled_rewards = scored_suffixes
            else:
                sampled_scores = scored_suffixes.correctness
                sampled_rewards = scored_suffixes.rewards
        else:
            sampled_scores = torch.empty(0, dtype=torch.float32)
            sampled_rewards = torch.empty(0, dtype=torch.float32)

        for request, score in zip(requests, sampled_scores, strict=True):
            if request.kind == "q":
                q_reward_sums[request.anchor_index] += score
                q_reward_counts[request.anchor_index] += 1
            else:
                v_reward_sums[request.anchor_index] += score
                v_reward_counts[request.anchor_index] += 1
        if torch.any(q_reward_counts == 0) or torch.any(v_reward_counts == 0):
            raise RuntimeError("Counterfactual suffix bookkeeping left an anchor without a Q or V estimate.")
        q_estimates = q_reward_sums / q_reward_counts
        v_estimates = v_reward_sums / v_reward_counts
        anchor_credit = q_estimates - v_estimates

        dense_targets = torch.zeros_like(step_uncertainty)
        dense_q_targets = torch.zeros_like(step_uncertainty)
        dense_v_targets = torch.zeros_like(step_uncertainty)
        dense_targets[anchor_mask] = anchor_credit
        dense_q_targets[anchor_mask] = q_estimates
        dense_v_targets[anchor_mask] = v_estimates
        batch.batch["credit_anchor_mask"] = anchor_mask
        batch.batch["credit_anchor_targets"] = dense_targets
        batch.batch["credit_anchor_q_targets"] = dense_q_targets
        batch.batch["credit_anchor_v_targets"] = dense_v_targets
        batch.batch["credit_start_value_mask"] = torch.ones(len(batch), dtype=torch.bool)
        batch.batch["credit_start_value_train_mask"] = initial_state_train_mask
        batch.batch["credit_start_value_targets"] = initial_state_values
        batch.batch["credit_terminal_value_mask"] = torch.ones(len(batch), dtype=torch.bool)
        batch.batch["credit_terminal_value_targets"] = original_scores
        metrics = {
            "credit/mc_mean": float(anchor_credit.mean().item()),
            "credit/mc_abs_mean": float(anchor_credit.abs().mean().item()),
            "credit/q_mean": float(q_estimates.mean().item()),
            "credit/v_mean": float(v_estimates.mean().item()),
        }
        training_branches = None
        if bool(self.config.train_mc_branches):
            grouped_requests: dict[tuple[int, str], list[int]] = {}
            for request_index, request in enumerate(requests):
                grouped_requests.setdefault((request.anchor_index, request.kind), []).append(request_index)
            candidates = []
            for (anchor_index, kind), request_indices in grouped_requests.items():
                expected_size = int(self.config.num_samples)
                if len(request_indices) != expected_size or any(not continuations[index] for index in request_indices):
                    continue
                source_index = requests[request_indices[0]].source_index
                candidate_correctness = sampled_scores[request_indices].float()
                candidates.append(
                    _BranchCandidate(
                        prompt_uid=uids[source_index],
                        anchor_index=anchor_index,
                        source_index=source_index,
                        kind=kind,
                        response_prefix=requests[request_indices[0]].response_prefix,
                        anchor_entropy=float(anchor_entropies[anchor_index].item()),
                        correctness_variance=float(candidate_correctness.var(unbiased=False).item()),
                        request_indices=tuple(request_indices),
                    )
                )
            selected_groups = self._select_training_branch_groups(
                candidates,
                uids,
                groups_per_prompt=int(self.config.branch_groups_per_prompt),
                seed=int(self.config.selection_seed),
                global_step=global_step,
            )
            training_branches = self._make_training_branch_batch(
                batch,
                requests,
                full_responses,
                continuations,
                sampled_scores,
                sampled_rewards,
                q_estimates,
                v_estimates,
                selected_groups,
                global_step=global_step,
            )
        return CounterfactualCreditRolloutResult(metrics=metrics, training_branches=training_branches)


__all__ = ["CounterfactualCreditRolloutResult", "SparseCounterfactualCreditSupervisor"]
