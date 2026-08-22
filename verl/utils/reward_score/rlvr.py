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

"""math_verify reward for the unified ``rlvr`` data source.

With the DAPO reward manager, each synchronous call runs in the main thread of
a Ray actor process, where math_verify's native POSIX timeouts work without an
additional process pool.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

_GOLD_EXTRACTION_CONFIG = (LatexExtractionConfig(),)
_PRED_EXTRACTION_CONFIG = (ExprExtractionConfig(), LatexExtractionConfig())
_ANSWER_LINE_PATTERN = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:Final\s+)?Answer(?:\*\*)?\s*:\s*(.*?)\s*$"
)
_MAX_SOLUTION_CHARS = 4096


@lru_cache(maxsize=4096)
def _parse_ground_truth(ground_truth: str) -> list[Any]:
    ground_truth = ground_truth.strip()
    if ground_truth.startswith("\\boxed{"):
        gold_text = ground_truth
    else:
        gold_text = "\\boxed{" + ground_truth + "}"
    return parse(
        gold_text,
        extraction_config=_GOLD_EXTRACTION_CONFIG,
        parsing_timeout=5,
    )


def _extract_answer_candidate(solution_str: str) -> str | None:
    """Return everything after the last explicit answer label.

    Keeping the suffix rather than only one line supports multiline LaTeX such
    as matrices and ``\\text{...}`` answers containing an embedded newline.
    """
    for match in reversed(list(_ANSWER_LINE_PATTERN.finditer(solution_str))):
        candidate = solution_str[match.start(1) :].strip()
        if not candidate:
            continue
        if candidate.startswith("**") and candidate.endswith("**") and len(candidate) > 4:
            candidate = candidate[2:-2].strip()
        if candidate.startswith("$") and not candidate.endswith("$"):
            candidate = candidate[1:].strip()
        if candidate.endswith("."):
            candidate = candidate[:-1].rstrip()
        return candidate or None
    return None


def _parse_prediction(solution_str: str) -> list[Any]:
    solution_tail = solution_str[-_MAX_SOLUTION_CHARS:]
    candidate = _extract_answer_candidate(solution_tail)
    if candidate:
        if candidate.startswith(("$", "\\(", "\\[", "\\boxed{", "\\fbox{")):
            candidate_text = candidate
        else:
            candidate_text = "\\boxed{" + candidate + "}"
        prediction = parse(
            candidate_text,
            extraction_config=_PRED_EXTRACTION_CONFIG,
            parsing_timeout=5,
        )
        if prediction:
            return prediction

    return parse(
        solution_tail,
        extraction_config=_PRED_EXTRACTION_CONFIG,
        parsing_timeout=5,
    )


def verify_solution(solution_str: str, ground_truth: str) -> bool:
    """Return whether the generated final answer matches the ground truth."""
    try:
        gold = _parse_ground_truth(ground_truth)
        prediction = _parse_prediction(solution_str)
        if not gold or not prediction:
            return False
        return bool(verify(gold, prediction, timeout_seconds=5))
    except Exception:
        return False


def compute_score(solution_str: str, ground_truth: str) -> dict[str, Any]:
    """Score a generated solution using math_verify equivalence.

    The return shape intentionally matches ``math_dapo.compute_score``.
    """
    correct = verify_solution(solution_str, ground_truth)
    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
    }
