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

"""math_verify-based reward for the unified ``rlvr`` math data source."""

from __future__ import annotations

import multiprocessing
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import Any

from math_verify import parse, verify

ANSWER_LINE_PATTERN = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:Final\s+)?Answer(?:\*\*)?\s*:\s*(.*?)\s*$"
)
BOX_COMMANDS = ("\\boxed{", "\\fbox{")

_PROCESS_POOL_WORKERS = 4
_PROCESS_POOL: ProcessPoolExecutor | None = None
_PROCESS_POOL_LOCK = threading.Lock()


def _parse_math(text: str) -> list[Any]:
    """Parse math with math_verify's bounded, process-safe timeout."""
    return parse(text, parsing_timeout=5)


def extract_answer_candidate(solution_str: str) -> str | None:
    """Extract the last non-empty explicit Answer line, if one exists."""
    matches = ANSWER_LINE_PATTERN.findall(solution_str)
    for match in reversed(matches):
        candidate = match.strip()
        if candidate:
            return _strip_outer_formatting(candidate)
    return None


def extract_last_boxed(solution_str: str) -> str | None:
    """Extract the last balanced ``\boxed{...}`` or ``\fbox{...}``."""
    starts = [solution_str.rfind(command) for command in BOX_COMMANDS]
    start = max(starts)
    if start < 0:
        return None

    opening_brace = solution_str.find("{", start)
    depth = 0
    for index in range(opening_brace, len(solution_str)):
        character = solution_str[index]
        if character == "{" and (index == 0 or solution_str[index - 1] != "\\"):
            depth += 1
        elif character == "}" and (index == 0 or solution_str[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return solution_str[start : index + 1]
    return None


def _strip_outer_formatting(candidate: str) -> str:
    candidate = candidate.strip()
    if candidate.startswith("**") and candidate.endswith("**") and len(candidate) > 4:
        candidate = candidate[2:-2].strip()

    delimiter_pairs = (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"))
    for left, right in delimiter_pairs:
        if candidate.startswith(left) and candidate.endswith(right) and len(candidate) > len(left) + len(right):
            candidate = candidate[len(left) : -len(right)].strip()
            break

    # DAPO-style generations often contain an unmatched opening dollar sign.
    if candidate.startswith("$"):
        candidate = candidate[1:].strip()
    if candidate.endswith("."):
        candidate = candidate[:-1].rstrip()
    return candidate


@lru_cache(maxsize=4096)
def _parse_ground_truth(ground_truth: str) -> list[Any]:
    ground_truth = ground_truth.strip()
    if ground_truth.startswith("\\boxed{"):
        gold_text = ground_truth
    else:
        gold_text = "\\boxed{" + ground_truth + "}"
    return _parse_math(gold_text)


def _parse_prediction(solution_str: str) -> tuple[list[Any], str | None]:
    candidate = extract_answer_candidate(solution_str)
    if candidate:
        if candidate.startswith("\\boxed{"):
            prediction = _parse_math(candidate)
        else:
            # Boxing makes bare LaTeX intervals, matrices, text, and symbolic
            # expressions parseable while retaining math_verify semantics.
            prediction = _parse_math("\\boxed{" + candidate + "}")
        if prediction:
            return prediction, candidate

    boxed = extract_last_boxed(solution_str)
    if boxed:
        prediction = _parse_math(boxed)
        if prediction:
            return prediction, boxed

    prediction = _parse_math(solution_str)
    prediction_text = str(prediction[0]) if prediction else candidate
    return prediction, prediction_text


def _verify_solution_local(solution_str: str, ground_truth: str) -> tuple[bool, str | None]:
    try:
        gold = _parse_ground_truth(ground_truth)
        prediction, prediction_text = _parse_prediction(solution_str)
        if not gold or not prediction:
            return False, prediction_text
        return bool(verify(gold, prediction, timeout_seconds=5)), prediction_text
    except Exception:
        return False, None


def _get_process_pool() -> ProcessPoolExecutor:
    """Return a per-reward-worker pool whose workers can use signal timeouts."""
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        with _PROCESS_POOL_LOCK:
            if _PROCESS_POOL is None:
                _PROCESS_POOL = ProcessPoolExecutor(
                    max_workers=_PROCESS_POOL_WORKERS,
                    mp_context=multiprocessing.get_context("spawn"),
                    max_tasks_per_child=1000,
                )
    return _PROCESS_POOL


def verify_solution(solution_str: str, ground_truth: str) -> tuple[bool, str | None]:
    """Return equivalence without running signal-based timeouts in a thread."""
    if threading.current_thread() is threading.main_thread():
        return _verify_solution_local(solution_str, ground_truth)

    try:
        return _get_process_pool().submit(
            _verify_solution_local,
            solution_str,
            ground_truth,
        ).result()
    except Exception:
        return False, None


def compute_score(solution_str: str, ground_truth: str) -> dict[str, Any]:
    """Score a generated solution using math_verify equivalence.

    The return shape intentionally matches ``math_dapo.compute_score``.
    """
    correct, prediction = verify_solution(solution_str, ground_truth)
    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": prediction,
    }
