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
"""Token-aligned, structure-aware reasoning-step splitting utilities."""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

# Named markers are accepted at the beginning of a line. Ordinary numbered
# markers are deliberately weaker: they upgrade an existing paragraph break
# but do not split every equation/list item introduced by a single newline.
STEP_MARKER_RE = re.compile(
    r"""
    \A[ \t]*
    (?:\#{1,6}[ \t]*)?
    (?:\*{1,2}[ \t]*)?
    (?:step|步骤)
    [ \t]*(?:\*{1,2}[ \t]*)?
    (?:[\#：:\-][ \t]*)?
    (?P<number>\d+|[一二三四五六七八九十百]+)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

CASE_MARKER_RE = re.compile(
    r"""
    \A[ \t]*
    (?:\#{1,6}[ \t]*)?
    (?:\*{1,2}[ \t]*)?
    (?:case|情况)
    [ \t]*(?:\*{1,2}[ \t]*)?
    (?:[\#：:\-][ \t]*)?
    (?:\d+|[一二三四五六七八九十百]+)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

FINAL_ANSWER_RE = re.compile(
    r"""
    \A[ \t]*
    (?:\#{1,6}[ \t]*)?
    (?:\*{1,2}[ \t]*)?
    (?:final[ \t]+(?:answer|result)|最终答案|答案|结论)
    (?:\*{1,2})?[ \t]*(?:[：:]|\Z)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

ORDINARY_NUMBER_MARKER_RE = re.compile(
    r"""
    \A[ \t]*
    (?:\#{1,6}[ \t]*)?
    (?:\*{1,2}[ \t]*)?
    (?:[(（][ \t]*)?
    (?:
        [1-9]\d*[ \t]*(?:[)）]|[.．](?!\d)|、)
        |
        [A-Z][.．]
        |
        [一二三四五六七八九十百]+、
    )
    """,
    flags=re.VERBOSE,
)

_PARAGRAPH_BREAK_RE = re.compile(r"(?:\r?\n[ \t]*){2,}")
_LINE_START_RE = re.compile(r"(?m)^")
_SENTENCE_END_RE = re.compile(r"[.!?;。！？；](?:[\"'”’）)\]}】]*)(?=[ \t\r\n]|$)")
_SINGLE_NEWLINE_RE = re.compile(r"\r?\n[ \t]*")
_MATH_BLOCK_START_RE = re.compile(r"\A(?:\$\$|\\\[|\\begin\{)")

_BOUNDARY_PRIORITY = {
    "line_fallback": 1,
    "sentence_fallback": 2,
    "paragraph": 3,
    "explicit": 4,
    "final_answer": 5,
    "forced": 0,
}

_SINGLE_TOKEN_TEXT_CACHE: dict[int, tuple[Any, dict[int, str]]] = {}


@dataclass(frozen=True)
class StepSpan:
    """One half-open token span and the reason its right boundary was chosen."""

    start: int
    end: int
    end_boundary_type: str

    @property
    def length(self) -> int:
        return self.end - self.start


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


def _decode_token_pieces(tokenizer: Any, input_ids: list[int]) -> list[str]:
    """Decode original actions individually so every text boundary maps to them.

    Structural characters used by this splitter (newlines, ASCII markers and
    punctuation) have stable single-token decodings in the supported text
    tokenizers. Keeping the original pieces also avoids the incorrect but
    tempting decode-then-retokenize round trip.
    """

    tokenizer_key = id(tokenizer)
    cache_entry = _SINGLE_TOKEN_TEXT_CACHE.get(tokenizer_key)
    if cache_entry is None or cache_entry[0] is not tokenizer:
        token_text_cache: dict[int, str] = {}
        _SINGLE_TOKEN_TEXT_CACHE[tokenizer_key] = (tokenizer, token_text_cache)
    else:
        token_text_cache = cache_entry[1]

    pieces = []
    for token_id in input_ids:
        if token_id not in token_text_cache:
            token_text_cache[token_id] = _decode(tokenizer, [token_id], response_index=-1)
        pieces.append(token_text_cache[token_id])

    # Byte-level tokenizers can split one UTF-8 character across actions. A
    # single-token decode then contains a temporary replacement character even
    # though decoding the short run is lossless. Assign the resolved text to
    # the final token in that run; empty preceding pieces retain every original
    # token boundary without inventing a retokenized sequence.
    token_index = 0
    while token_index < len(pieces):
        if "\ufffd" not in pieces[token_index]:
            token_index += 1
            continue
        resolved_end = None
        resolved_text = None
        for token_end in range(token_index + 2, min(len(input_ids), token_index + 8) + 1):
            candidate_text = _decode(tokenizer, input_ids[token_index:token_end], response_index=-1)
            if "\ufffd" not in candidate_text:
                resolved_end = token_end
                resolved_text = candidate_text
                break
        if resolved_end is None:
            token_index += 1
            continue
        pieces[token_index:resolved_end] = [""] * (resolved_end - token_index - 1) + [resolved_text]
        token_index = resolved_end
    return pieces


def _protected_ranges(text: str) -> list[tuple[int, int]]:
    """Return code and LaTeX ranges whose interiors must not be split."""

    patterns = (
        re.compile(r"(?P<fence>`{3,}|~{3,})[^\n]*(?:\n|\Z).*?(?:(?P=fence)|\Z)", re.DOTALL),
        re.compile(r"\$\$.*?(?:\$\$|\Z)", re.DOTALL),
        re.compile(r"\\\[.*?(?:\\\]|\Z)", re.DOTALL),
        re.compile(
            r"\\begin\{(?P<env>aligned\*?|align\*?|equation\*?|gathered|gather\*?|cases|"
            r"[bBpvV]?matrix)\}.*?\\end\{(?P=env)\}",
            re.DOTALL,
        ),
    )
    ranges = sorted((match.start(), match.end()) for pattern in patterns for match in pattern.finditer(text))
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _inside_protected_range(position: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < position < end for start, end in ranges)


def _snap_to_token_boundary(character_position: int, token_character_ends: list[int]) -> int:
    """Map a text position to the nearest boundary between original tokens."""

    right = bisect.bisect_left(token_character_ends, character_position)
    if right <= 0:
        return 0
    if right >= len(token_character_ends):
        return len(token_character_ends) - 1
    left_distance = character_position - token_character_ends[right - 1]
    right_distance = token_character_ends[right] - character_position
    return right - 1 if left_distance < right_distance else right


def _block_marker_type(text: str, *, allow_ordinary: bool) -> str | None:
    if FINAL_ANSWER_RE.match(text) is not None:
        return "final_answer"
    if STEP_MARKER_RE.match(text) is not None or CASE_MARKER_RE.match(text) is not None:
        return "explicit"
    if allow_ordinary and ORDINARY_NUMBER_MARKER_RE.match(text) is not None:
        return "explicit"
    return None


def _is_marker_only_heading(line: str) -> bool:
    """Whether a line is a heading that should stay with its following body."""

    stripped = line.strip()
    if _block_marker_type(stripped, allow_ordinary=True) is None:
        return False
    return stripped.startswith("#") or (stripped.startswith("**") and stripped.endswith("**"))


def _add_candidate(candidates: dict[int, str], position: int, kind: str, *, num_tokens: int) -> None:
    if position <= 0 or position >= num_tokens:
        return
    previous = candidates.get(position)
    if previous is None or _BOUNDARY_PRIORITY[kind] > _BOUNDARY_PRIORITY[previous]:
        candidates[position] = kind


def _candidate_boundaries(
    text: str,
    token_character_ends: list[int],
) -> tuple[dict[int, str], dict[int, str]]:
    """Collect structural boundaries and weak fallbacks at token positions."""

    num_tokens = len(token_character_ends) - 1
    protected = _protected_ranges(text)
    structural: dict[int, str] = {}
    fallback: dict[int, str] = {}

    for match in _PARAGRAPH_BREAK_RE.finditer(text):
        character_position = match.end()
        if _inside_protected_range(character_position, protected):
            continue
        previous_line_start = text.rfind("\n", 0, match.start()) + 1
        if _is_marker_only_heading(text[previous_line_start : match.start()]):
            continue
        right_text = text[character_position : character_position + 256]
        # A displayed equation normally belongs to its preceding explanation;
        # the paragraph break after the equation remains available.
        if _MATH_BLOCK_START_RE.match(right_text.lstrip(" \t")) is not None:
            continue
        kind = _block_marker_type(right_text, allow_ordinary=True) or "paragraph"
        token_position = _snap_to_token_boundary(character_position, token_character_ends)
        _add_candidate(structural, token_position, kind, num_tokens=num_tokens)

    # Named steps/cases/final answers remain explicit with only a single line
    # break. Ordinary numbered lists require a paragraph break to avoid turning
    # every formula list into reasoning steps.
    for match in _LINE_START_RE.finditer(text):
        character_position = match.start()
        if character_position == 0 or _inside_protected_range(character_position, protected):
            continue
        kind = _block_marker_type(text[character_position : character_position + 256], allow_ordinary=False)
        if kind is None:
            continue
        token_position = _snap_to_token_boundary(character_position, token_character_ends)
        _add_candidate(structural, token_position, kind, num_tokens=num_tokens)

    # A completion has only one semantic transition into its final answer.
    # Degenerate generations sometimes repeat the same final-answer line until
    # truncation; treating every repetition as an exempt short explicit step
    # would let one response contribute hundreds of meaningless candidates.
    final_answer_positions = sorted(position for position, kind in structural.items() if kind == "final_answer")
    for position in final_answer_positions[1:]:
        structural[position] = "paragraph"

    for match in _SENTENCE_END_RE.finditer(text):
        character_position = match.end()
        if _inside_protected_range(character_position, protected):
            continue
        token_position = _snap_to_token_boundary(character_position, token_character_ends)
        _add_candidate(fallback, token_position, "sentence_fallback", num_tokens=num_tokens)

    for match in _SINGLE_NEWLINE_RE.finditer(text):
        character_position = match.end()
        if _inside_protected_range(character_position, protected):
            continue
        token_position = _snap_to_token_boundary(character_position, token_character_ends)
        _add_candidate(fallback, token_position, "line_fallback", num_tokens=num_tokens)

    return structural, fallback


def _merge_short_steps(candidates: dict[int, str], *, num_tokens: int, min_step_tokens: int) -> dict[int, str]:
    """Merge ordinary short fragments while preserving explicit short steps."""

    selected = dict(candidates)
    while selected:
        positions = [0, *sorted(selected), num_tokens]
        removed = False
        for span_index, (start, end) in enumerate(zip(positions, positions[1:], strict=False)):
            if end - start >= min_step_tokens:
                continue
            start_kind = selected.get(start)
            if start_kind in {"explicit", "final_answer"}:
                continue
            if span_index == 0:
                # Merge a short introductory fragment into the first real step.
                selected.pop(end, None)
            else:
                # An ordinary short paragraph is a continuation of its left step.
                selected.pop(start, None)
            removed = True
            break
        if not removed:
            break
    return selected


def _split_long_steps(
    candidates: dict[int, str],
    fallback_candidates: dict[int, str],
    *,
    num_tokens: int,
    min_step_tokens: int,
    max_step_tokens: int,
) -> dict[int, str]:
    """Bound long spans at the latest safe sentence/newline before the limit."""

    selected = dict(candidates)
    original_positions = [0, *sorted(selected), num_tokens]
    safe_positions = sorted(set(fallback_candidates) | set(candidates))

    for original_start, original_end in zip(original_positions, original_positions[1:], strict=False):
        start = original_start
        while original_end - start > max_step_tokens:
            upper = min(start + max_step_tokens, original_end - min_step_tokens)
            lower = start + min_step_tokens
            safe_index = bisect.bisect_right(safe_positions, upper) - 1
            if safe_index >= 0 and safe_positions[safe_index] >= lower:
                split_position = safe_positions[safe_index]
                kind = candidates.get(split_position, fallback_candidates.get(split_position, "sentence_fallback"))
            else:
                split_position = upper
                kind = "forced"
            _add_candidate(selected, split_position, kind, num_tokens=num_tokens)
            start = split_position

    return selected


def split_token_ids(
    tokenizer: Any,
    input_ids: Sequence[int],
    *,
    min_step_tokens: int = 96,
    max_step_tokens: int = 512,
) -> list[StepSpan]:
    """Split one nonempty original token sequence into half-open step spans.

    Paragraphs and explicit step markers provide normal boundaries. Short
    ordinary fragments are merged, while long spans fall back to sentence or
    line endings and finally to a hard token boundary.
    """

    if not isinstance(min_step_tokens, int) or isinstance(min_step_tokens, bool) or min_step_tokens < 1:
        raise ValueError(f"min_step_tokens must be an integer >= 1, got {min_step_tokens!r}")
    if not isinstance(max_step_tokens, int) or isinstance(max_step_tokens, bool) or max_step_tokens < min_step_tokens:
        raise ValueError(
            f"max_step_tokens must be an integer >= min_step_tokens ({min_step_tokens}), got {max_step_tokens!r}"
        )

    ids = [int(token_id) for token_id in input_ids]
    if not ids:
        raise ValueError("input_ids must contain at least one token")

    pieces = _decode_token_pieces(tokenizer, ids)
    token_character_ends = [0]
    for piece in pieces:
        token_character_ends.append(token_character_ends[-1] + len(piece))
    text = "".join(pieces)

    structural, fallback = _candidate_boundaries(text, token_character_ends)
    selected = _merge_short_steps(structural, num_tokens=len(ids), min_step_tokens=min_step_tokens)
    selected = _split_long_steps(
        selected,
        fallback,
        num_tokens=len(ids),
        min_step_tokens=min_step_tokens,
        max_step_tokens=max_step_tokens,
    )

    positions = [0, *sorted(selected), len(ids)]
    spans = [
        StepSpan(start=start, end=end, end_boundary_type=selected.get(end, "response_end"))
        for start, end in zip(positions, positions[1:], strict=False)
    ]
    if any(span.length <= 0 for span in spans):
        raise RuntimeError("Step splitter created an empty token span")
    return spans


def build_step_end_mask(
    tokenizer: Any,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    min_step_tokens: int = 96,
    max_step_tokens: int = 512,
) -> torch.Tensor:
    """Return a boolean mask selecting the final original token of every step."""

    if not isinstance(responses, torch.Tensor) or not isinstance(response_mask, torch.Tensor):
        raise TypeError("responses and response_mask must be torch.Tensor instances")
    if responses.ndim != 2 or response_mask.ndim != 2:
        raise ValueError(f"responses and response_mask must be rank-2, got {responses.ndim} and {response_mask.ndim}")
    if responses.shape != response_mask.shape:
        raise ValueError(
            f"responses and response_mask must have the same shape, got "
            f"{tuple(responses.shape)} and {tuple(response_mask.shape)}"
        )

    eos_token_ids = _normalize_eos_token_ids(tokenizer)
    responses_cpu = responses.detach().cpu()
    active_mask = response_mask.detach().to(device="cpu", dtype=torch.bool)
    step_end_mask_cpu = torch.zeros_like(responses_cpu, dtype=torch.bool)

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
            # Immediate EOS is a valid empty completion and remains one action.
            step_end_mask_cpu[response_index, first_active_position] = True
            continue

        spans = split_token_ids(
            tokenizer,
            input_ids,
            min_step_tokens=min_step_tokens,
            max_step_tokens=max_step_tokens,
        )
        for span in spans:
            step_end_mask_cpu[response_index, active_positions[span.end - 1]] = True

    return step_end_mask_cpu.to(device=responses.device)


def build_step_start_mask(step_end_mask: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Derive one first-token marker for every shared step endpoint."""

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
    "CASE_MARKER_RE",
    "FINAL_ANSWER_RE",
    "ORDINARY_NUMBER_MARKER_RE",
    "STEP_MARKER_RE",
    "StepSpan",
    "build_step_end_mask",
    "build_step_start_mask",
    "split_token_ids",
]
