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

import time

from verl.workers.actor.dp_actor import DataParallelPPOActor


class CharTokenizer:
    """Minimal tokenizer that maps each character to one token."""

    @staticmethod
    def decode(token_ids, **_) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


STRONG_BOUNDARY_PATTERNS = [
    r"(?i)\bStep\s*\d+\b",
    r"\b\d+\.\s",
    (
        r"(?i)^(?:[#*_]+[ \t]+)*[#*_]*[ \t]*"
        r"(?:First|Firstly|Second|Secondly|Third|Thirdly|Next|Then|Finally|Similarly)\b"
    ),
]


def delimiter_end_positions(text: str) -> list[int]:
    positions = []
    start = 0
    while (delimiter_start := text.find("\n\n", start)) >= 0:
        positions.append(delimiter_start + 1)
        start = delimiter_start + 2
    return positions


def select_boundaries(
    text: str,
    *,
    marker_filter: bool = True,
    lookahead: int = 32,
    fallback_min_tokens: int = 128,
    max_steps_per_response: int = 0,
) -> list[int]:
    token_ids = [ord(char) for char in text]
    return DataParallelPPOActor._select_step_end_positions(
        tokenizer=CharTokenizer(),
        valid_response_ids=token_ids,
        valid_response_positions=list(range(len(token_ids))),
        delimiter_token_ids=set(),
        delimiter_token_sequence=[ord("\n"), ord("\n")],
        delimiter="\n\n",
        step_interval=1,
        delimiter_step_marker_filter=marker_filter,
        delimiter_step_marker_lookahead=lookahead,
        delimiter_step_marker_patterns=STRONG_BOUNDARY_PATTERNS,
        delimiter_fallback_min_tokens=fallback_min_tokens,
        delimiter_max_steps_per_response=max_steps_per_response,
    )


def test_filter_disabled_keeps_every_delimiter():
    text = "short\n\nparagraph\n\nsequence"

    assert select_boundaries(text, marker_filter=False) == delimiter_end_positions(text)


def test_filter_keeps_only_strong_boundaries():
    segments = ["a" * 30, "Step 2: " + "b" * 8, "c" * 30, "d" * 10]
    text = "\n\n".join(segments)
    first, _, _ = delimiter_end_positions(text)

    assert select_boundaries(text, fallback_min_tokens=20) == [first]


def test_response_without_strong_boundary_uses_cumulative_token_fallback():
    segments = ["a" * 10, "b" * 10, "c" * 10, "d" * 10]
    text = "\n\n".join(segments)
    _, second, _ = delimiter_end_positions(text)

    assert select_boundaries(text, fallback_min_tokens=20) == [second]


def test_non_positive_fallback_interval_disables_fallback():
    text = "short\n\nparagraph\n\nsequence"

    assert select_boundaries(text, fallback_min_tokens=0) == []


def test_marker_within_lookahead_window_is_kept():
    text = "a" * 8 + "\n\n" + "### Step 2 appears"

    assert select_boundaries(text, lookahead=10) == delimiter_end_positions(text)


def test_marker_outside_lookahead_window_is_ignored():
    text = "a" * 8 + "\n\n" + "x" * 10 + "Step 2 appears"

    assert select_boundaries(text, lookahead=10) == []


def test_markdown_step_heading_is_a_strong_boundary():
    text = "brief reasoning\n\n### Step 2: continue"

    assert select_boundaries(text) == delimiter_end_positions(text)


def test_transition_word_is_not_a_strong_boundary():
    text = "brief reasoning\n\nTherefore, continue"

    assert select_boundaries(text) == []


def test_answer_prefix_is_not_a_strong_boundary():
    text = "brief reasoning\n\n### Final Answer 42"

    assert select_boundaries(text) == []


def test_numbered_list_at_paragraph_start_is_a_strong_boundary():
    text = "brief reasoning\n\n### 2. Continue"

    assert select_boundaries(text) == delimiter_end_positions(text)


def test_explicit_sequence_words_are_strong_boundaries():
    for marker in (
        "First",
        "Firstly",
        "Second",
        "Secondly",
        "Third",
        "Thirdly",
        "Next",
        "Then",
        "Finally",
        "Similarly",
    ):
        text = f"brief reasoning\n\n### **{marker}**, continue"

        assert select_boundaries(text) == delimiter_end_positions(text)


def test_sequence_word_inside_paragraph_is_not_a_strong_boundary():
    text = "brief reasoning\n\nContinue by considering the First case"

    assert select_boundaries(text) == []


def test_numbered_marker_inside_lookahead_is_a_strong_boundary():
    text = "brief reasoning\n\nContinue with 2. another argument"

    assert select_boundaries(text) == delimiter_end_positions(text)


def test_bare_step_words_and_chinese_markers_are_not_strong_boundaries():
    english = "brief reasoning\n\nNow begin the next Step carefully"
    chinese = "简短推理\n\n现在开始下一个步骤并验证"

    assert select_boundaries(english, fallback_min_tokens=0) == []
    assert select_boundaries(chinese, fallback_min_tokens=0) == []


def test_decimal_inside_lookahead_is_not_a_numbered_marker():
    text = "brief reasoning\n\nThe value is 2.5 and remains unchanged"

    assert select_boundaries(text) == []


def test_parenthesized_numbers_are_not_strong_boundaries():
    closing_parenthesis = "brief reasoning\n\nContinue with 1)another argument"
    parenthesized = "brief reasoning\n\nContinue with (1)another argument"

    assert select_boundaries(closing_parenthesis, fallback_min_tokens=0) == []
    assert select_boundaries(parenthesized, fallback_min_tokens=0) == []


def test_period_number_requires_trailing_whitespace():
    text = "brief reasoning\n\nContinue with 1.without a space"

    assert select_boundaries(text) == []


def test_markdown_marker_mismatch_does_not_backtrack_exponentially():
    text = "brief reasoning\n\n" + "#" * 22 + "X"
    start = time.monotonic()

    assert select_boundaries(text, lookahead=32) == []
    assert time.monotonic() - start < 0.25


def test_max_steps_evenly_keeps_first_and_last_boundaries():
    text = "intro\n\n" + "\n\n".join(f"Step {idx}: reasoning" for idx in range(1, 9))
    all_positions = delimiter_end_positions(text)

    selected = select_boundaries(text, max_steps_per_response=4)

    assert len(selected) == 4
    assert selected[0] == all_positions[0]
    assert selected[-1] == all_positions[-1]


def test_non_positive_max_steps_disables_cap():
    text = "intro\n\n" + "\n\n".join(f"Step {idx}: reasoning" for idx in range(1, 41))

    assert select_boundaries(text, max_steps_per_response=0) == delimiter_end_positions(text)
