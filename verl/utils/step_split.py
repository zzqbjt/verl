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
"""Token-aligned reasoning-step splitting utilities."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import torch

# A numeric index is intentional: phrases such as "solve step by step" are not
# explicit step markers.
STEP_MARKER_RE = re.compile(
    r"""
    (?:\#{1,6}\s*)?             # optional Markdown heading
    (?:\*{1,2}\s*)?             # optional opening emphasis
    (?:step|步骤)
    \s*(?:\*{1,2}\s*)?          # also accept: Step **1**
    (?:[\#：:\-]\s*)?
    (?P<number>\d+)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

ORDINARY_NUMBER_MARKER_RE = re.compile(
    r"""
    (?<![\w.])
    (?:
        (?P<arabic>[1-9]\d*)[ \t]*(?:[)）]|[.．](?!\d)|、)
        |
        (?P<latin>[A-Z])[.．]
        |
        (?P<chinese>[一二三四五六七八九十百]+)、
    )
    """,
    flags=re.VERBOSE,
)

_SINGLE_TOKEN_TEXT_CACHE: dict[int, tuple[Any, dict[int, str]]] = {}


def _normalize_eos_token_ids(tokenizer: Any) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, Sequence) and not isinstance(eos_token_id, str | bytes):
        return {int(token_id) for token_id in eos_token_id}
    return {int(eos_token_id)}


def _decode(tokenizer: Any, input_ids: list[int], *, response_index: int) -> str:
    text = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(text, str):
        raise TypeError(f"Response {response_index}: tokenizer.decode must return str, got {type(text).__name__}")
    return text


def _delimiter_token_ids(
    tokenizer: Any,
    token_ids: set[int],
    *,
    delimiter: str,
) -> set[int]:
    """Find original action tokens whose own decoded text contains a delimiter.

    The cache avoids a tokenizer call for every occurrence of a common token
    in long batches.
    """

    tokenizer_key = id(tokenizer)
    cache_entry = _SINGLE_TOKEN_TEXT_CACHE.get(tokenizer_key)
    if cache_entry is None or cache_entry[0] is not tokenizer:
        token_text_cache: dict[int, str] = {}
        _SINGLE_TOKEN_TEXT_CACHE[tokenizer_key] = (tokenizer, token_text_cache)
    else:
        token_text_cache = cache_entry[1]

    delimiter_ids = set()
    for token_id in token_ids:
        if token_id not in token_text_cache:
            token_text_cache[token_id] = _decode(tokenizer, [token_id], response_index=-1)
        if delimiter in token_text_cache[token_id]:
            delimiter_ids.add(token_id)
    return delimiter_ids


def _step_end_indices(
    tokenizer: Any,
    input_ids: list[int],
    delimiter_token_ids: set[int],
    *,
    lookahead_tokens: int,
    separate_preamble: bool,
    response_index: int,
) -> list[int]:
    detected_splits: list[tuple[int, int]] = []
    for token_index, token_id in enumerate(input_ids):
        if token_id not in delimiter_token_ids or token_index + 1 >= len(input_ids):
            continue

        lookahead_end = min(len(input_ids), token_index + 1 + lookahead_tokens)
        following_text = _decode(
            tokenizer,
            input_ids[token_index + 1 : lookahead_end],
            response_index=response_index,
        )
        explicit_match = STEP_MARKER_RE.search(following_text)
        ordinary_match = ORDINARY_NUMBER_MARKER_RE.search(following_text)
        marker_candidates = []
        if explicit_match is not None:
            marker_candidates.append((explicit_match.start(), int(explicit_match.group("number"))))
        if ordinary_match is not None:
            ordinary_groups = ordinary_match.groupdict()
            if ordinary_groups["arabic"] is not None:
                ordinary_number = int(ordinary_groups["arabic"])
            elif ordinary_groups["latin"] is not None:
                ordinary_number = ord(ordinary_groups["latin"]) - ord("A") + 1
            else:
                ordinary_number = 1 if ordinary_groups["chinese"] == "一" else 2
            marker_candidates.append((ordinary_match.start(), ordinary_number))

        if marker_candidates:
            _, marker_number = min(marker_candidates, key=lambda candidate: candidate[0])
            # Split after the double-newline token, so punctuation and the
            # delimiter stay in the preceding step.
            detected_splits.append((token_index + 1, marker_number))

    # More than one whitespace delimiter can see the same marker inside the
    # lookahead window.  Keep the delimiter closest to that marker instead of
    # creating an artificial whitespace-only step between the candidates.
    collapsed_splits: list[tuple[int, int]] = []
    for position, marker_number in detected_splits:
        if collapsed_splits:
            previous_position = collapsed_splits[-1][0]
            between_text = _decode(
                tokenizer,
                input_ids[previous_position:position],
                response_index=response_index,
            )
            if not between_text.strip():
                collapsed_splits[-1] = (position, marker_number)
                continue
        collapsed_splits.append((position, marker_number))

    # A response may begin with blank lines before its first visible marker.
    # Do not materialize those leading blanks as a standalone reasoning step.
    if collapsed_splits:
        leading_text = _decode(
            tokenizer,
            input_ids[: collapsed_splits[0][0]],
            response_index=response_index,
        )
        if not leading_text.strip():
            collapsed_splits = collapsed_splits[1:]

    split_positions = [position for position, _ in collapsed_splits]
    if not separate_preamble and collapsed_splits and collapsed_splits[0][1] == 1:
        # Merge introductory text into Step 1 instead of creating Step 0.
        split_positions = split_positions[1:]

    split_positions = sorted(set(split_positions))
    span_starts = [0, *split_positions]
    span_ends = [*split_positions, len(input_ids)]
    step_end_indices = []
    for token_start, token_end in zip(span_starts, span_ends, strict=True):
        if token_start >= token_end:
            raise RuntimeError("Step splitter created an empty token span")
        step_text = _decode(
            tokenizer,
            input_ids[token_start:token_end],
            response_index=response_index,
        )
        if not step_text.strip():
            if len(span_starts) == 1:
                # A non-EOS response consisting only of whitespace is still a
                # valid sampled action sequence. Treat it as one step.
                return [len(input_ids) - 1]
            raise RuntimeError(f"Response {response_index}: step splitter created an empty text step")
        step_end_indices.append(token_end - 1)
    return step_end_indices


def build_step_end_mask(
    tokenizer: Any,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    lookahead_tokens: int = 10,
    separate_preamble: bool = False,
) -> torch.Tensor:
    """Return a boolean mask selecting the last token of every reasoning step.

    Active response tokens are selected by ``response_mask``. Trailing EOS
    tokens are removed first. Boundaries are then detected directly on the
    original generated token IDs. This matters because sampled BPE token
    sequences are not necessarily
    the canonical tokenization produced by encoding their decoded text.

    A delimiter is accepted only when one original token's decoded text
    contains a literal double newline and the following ``lookahead_tokens``
    tokens contain either a numeric ``Step``/``步骤`` marker or an ordinary
    numbered marker. The delimiter belongs to the preceding step. The last
    non-EOS response token always ends the final step. If a valid completion
    immediately emits EOS, that EOS action is its sole step endpoint.
    """

    if not isinstance(responses, torch.Tensor) or not isinstance(response_mask, torch.Tensor):
        raise TypeError("responses and response_mask must be torch.Tensor instances")
    if responses.ndim != 2 or response_mask.ndim != 2:
        raise ValueError(f"responses and response_mask must be rank-2, got {responses.ndim} and {response_mask.ndim}")
    if responses.shape != response_mask.shape:
        raise ValueError(
            f"responses and response_mask must have the same shape, got "
            f"{tuple(responses.shape)} and {tuple(response_mask.shape)}"
        )
    if lookahead_tokens <= 0:
        raise ValueError(f"lookahead_tokens must be positive, got {lookahead_tokens}")

    eos_token_ids = _normalize_eos_token_ids(tokenizer)
    responses_cpu = responses.detach().cpu()
    active_mask = response_mask.detach().to(device="cpu", dtype=torch.bool)
    step_end_mask = torch.zeros_like(responses, dtype=torch.bool)
    unique_active_token_ids = torch.unique(responses_cpu[active_mask]).tolist()
    active_token_ids = {int(token_id) for token_id in unique_active_token_ids if int(token_id) not in eos_token_ids}
    delimiter_token_ids = _delimiter_token_ids(
        tokenizer,
        active_token_ids,
        delimiter="\n\n",
    )

    for response_index in range(responses.shape[0]):
        active_positions = torch.nonzero(active_mask[response_index], as_tuple=False).flatten().tolist()
        if not active_positions:
            raise ValueError(f"Response {response_index}: no active response tokens after applying response_mask")
        first_active_position = active_positions[0]
        input_ids = [int(token_id) for token_id in responses_cpu[response_index, active_positions].tolist()]

        while input_ids and input_ids[-1] in eos_token_ids:
            input_ids.pop()
            active_positions.pop()
        if not input_ids:
            # Immediate EOS is a valid empty completion.  It has no text token
            # to serve as a step tail, so use the active terminal action itself
            # as the sole endpoint instead of dropping the response.
            step_end_mask[response_index, first_active_position] = True
            continue

        for step_end_index in _step_end_indices(
            tokenizer,
            input_ids,
            delimiter_token_ids,
            lookahead_tokens=lookahead_tokens,
            separate_preamble=separate_preamble,
            response_index=response_index,
        ):
            step_end_mask[response_index, active_positions[step_end_index]] = True

    return step_end_mask


def build_step_start_mask(step_end_mask: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Derive one first-token marker for every shared step endpoint.

    This function does not detect or alter boundaries. It converts the exact
    partition represented by ``step_end_mask`` into the corresponding starts,
    so providers that need a full step span reuse the same split result.
    """

    if not isinstance(step_end_mask, torch.Tensor) or not isinstance(response_mask, torch.Tensor):
        raise TypeError("step_end_mask and response_mask must be torch.Tensor instances")
    if step_end_mask.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("step_end_mask and response_mask must both be rank-2")
    if step_end_mask.shape != response_mask.shape:
        raise ValueError(
            "step_end_mask and response_mask must have the same shape, got "
            f"{tuple(step_end_mask.shape)} and {tuple(response_mask.shape)}"
        )

    ends = step_end_mask.detach().cpu().bool()
    active = response_mask.detach().cpu().bool()
    if torch.any(ends & ~active):
        raise ValueError("Step endpoints may select only active response tokens")

    starts = torch.zeros_like(ends)
    for row in range(ends.shape[0]):
        active_positions = torch.nonzero(active[row], as_tuple=False).flatten().tolist()
        end_positions = torch.nonzero(ends[row], as_tuple=False).flatten().tolist()
        if not active_positions:
            raise ValueError(f"Response {row}: response_mask contains no active token")
        if not end_positions:
            raise ValueError(f"Response {row}: step_end_mask contains no endpoint")

        active_rank = {position: rank for rank, position in enumerate(active_positions)}
        next_start_rank = 0
        for end_position in end_positions:
            if end_position not in active_rank:
                raise ValueError(f"Response {row}: endpoint {end_position} is not active")
            end_rank = active_rank[end_position]
            if end_rank < next_start_rank:
                raise ValueError(f"Response {row}: endpoints do not define ordered nonempty steps")
            starts[row, active_positions[next_start_rank]] = True
            next_start_rank = end_rank + 1

    if not torch.equal(starts.sum(dim=-1), ends.sum(dim=-1)):
        raise RuntimeError("Derived step starts do not align one-to-one with step endpoints")
    return starts.to(device=step_end_mask.device)


__all__ = [
    "ORDINARY_NUMBER_MARKER_RE",
    "STEP_MARKER_RE",
    "build_step_end_mask",
    "build_step_start_mask",
]
