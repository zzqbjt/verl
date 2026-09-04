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

import pytest
import torch

from verl.utils.step_split import build_step_end_mask, build_step_start_mask, split_token_ids


class PieceTokenizer:
    """Small reversible tokenizer whose IDs each cover one configured piece."""

    def __init__(self, pieces, *, eos_token_id=98, pad_token_id=99):
        self.pieces = dict(pieces)
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def decode(self, input_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(self.pieces[token_id] for token_id in input_ids)


def _endpoint_indices(tokenizer, token_ids, **kwargs):
    responses = torch.tensor([token_ids])
    mask = build_step_end_mask(tokenizer, responses, torch.ones_like(responses), **kwargs)
    return mask.nonzero(as_tuple=False).tolist()


def test_step_starts_are_derived_from_the_shared_endpoint_partition():
    response_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0]])
    step_end_mask = torch.tensor(
        [[0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0]],
        dtype=torch.bool,
    )

    starts = build_step_start_mask(step_end_mask, response_mask)

    expected = torch.tensor(
        [[0, 1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0]],
        dtype=torch.bool,
    )
    torch.testing.assert_close(starts, expected)


def test_plain_paragraphs_are_steps_without_numeric_markers():
    tokenizer = PieceTokenizer({0: "First", 1: "\n", 2: "\n", 3: "Second"})

    endpoints = _endpoint_indices(tokenizer, [0, 1, 2, 3], min_step_tokens=1)

    assert endpoints == [[0, 2], [0, 3]]


def test_double_newline_inside_one_token_is_also_a_paragraph_boundary():
    tokenizer = PieceTokenizer({0: "First.\n\n", 1: "Second."})

    endpoints = _endpoint_indices(tokenizer, [0, 1], min_step_tokens=1)

    assert endpoints == [[0, 0], [0, 1]]


def test_named_step_marker_after_a_single_newline_is_explicit():
    tokenizer = PieceTokenizer({0: "First.", 1: "\n", 2: "### Step 2: next"})

    spans = split_token_ids(tokenizer, [0, 1, 2], min_step_tokens=1)

    assert [(span.start, span.end, span.end_boundary_type) for span in spans] == [
        (0, 2, "explicit"),
        (2, 3, "response_end"),
    ]


def test_case_marker_after_a_single_newline_is_explicit():
    tokenizer = PieceTokenizer({0: "First.", 1: "\n", 2: "Case 2: next"})

    endpoints = _endpoint_indices(tokenizer, [0, 1, 2], min_step_tokens=1)

    assert endpoints == [[0, 1], [0, 2]]


def test_ordinary_numbered_list_needs_a_paragraph_break():
    tokenizer = PieceTokenizer({0: "First.", 1: "\n", 2: "2. formula", 3: "\n\n", 4: "3. reasoning"})

    endpoints = _endpoint_indices(tokenizer, [0, 1, 2, 3, 4], min_step_tokens=1)

    assert endpoints == [[0, 3], [0, 4]]


def test_short_ordinary_paragraph_is_merged_to_the_left():
    tokenizer = PieceTokenizer({0: "A ", 1: "\n\n", 2: "tiny", 3: "\n\n", 4: "B "})
    token_ids = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]

    spans = split_token_ids(tokenizer, token_ids, min_step_tokens=4, max_step_tokens=20)

    assert [(span.start, span.end) for span in spans] == [(0, 7), (7, 11)]


def test_short_intro_is_merged_into_the_first_explicit_step():
    tokenizer = PieceTokenizer({0: "Intro", 1: "\n", 2: "Step 1: body", 3: " body"})

    spans = split_token_ids(tokenizer, [0, 1, 2, 3, 3], min_step_tokens=3, max_step_tokens=20)

    assert [(span.start, span.end) for span in spans] == [(0, 5)]


def test_short_explicit_step_is_preserved():
    tokenizer = PieceTokenizer({0: "body ", 1: "\n", 2: "Step 2: short", 3: "\n\n", 4: "next "})
    token_ids = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]

    spans = split_token_ids(tokenizer, token_ids, min_step_tokens=4, max_step_tokens=20)

    assert [(span.start, span.end) for span in spans] == [(0, 5), (5, 7), (7, 11)]


@pytest.mark.parametrize("heading", ["### Step 2: Calculate x", "**Step 2: Calculate x**"])
def test_marker_only_heading_stays_with_its_following_body(heading):
    tokenizer = PieceTokenizer({0: "First", 1: "\n", 2: heading, 3: "\n\n", 4: "body"})

    spans = split_token_ids(tokenizer, [0, 1, 2, 3, 4], min_step_tokens=1, max_step_tokens=20)

    assert [(span.start, span.end) for span in spans] == [(0, 2), (2, 5)]


def test_short_final_answer_is_preserved():
    tokenizer = PieceTokenizer({0: "reason ", 1: "\n", 2: "Final Answer: 42"})
    token_ids = [0, 0, 0, 0, 1, 2]

    spans = split_token_ids(tokenizer, token_ids, min_step_tokens=4, max_step_tokens=20)

    assert [(span.start, span.end) for span in spans] == [(0, 5), (5, 6)]


def test_repeated_final_answers_do_not_create_many_short_steps():
    tokenizer = PieceTokenizer({0: "reason ", 1: "\n\n", 2: "Final Answer: 42"})
    token_ids = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2]

    spans = split_token_ids(tokenizer, token_ids, min_step_tokens=4, max_step_tokens=20)

    assert [(span.start, span.end, span.end_boundary_type) for span in spans] == [
        (0, 5, "final_answer"),
        (5, 10, "response_end"),
    ]


def test_paragraph_before_display_math_does_not_detach_the_formula():
    tokenizer = PieceTokenizer(
        {
            0: "Explanation.",
            1: "\n\n",
            2: "\\[\na = b\n\\]",
            3: "\n\n",
            4: "Next argument.",
        }
    )

    endpoints = _endpoint_indices(tokenizer, [0, 1, 2, 3, 4], min_step_tokens=1)

    assert endpoints == [[0, 3], [0, 4]]


def test_blank_lines_inside_display_math_are_not_split():
    tokenizer = PieceTokenizer({0: "Before ", 1: "$$a", 2: "\n\n", 3: "b$$", 4: " after"})

    endpoints = _endpoint_indices(tokenizer, [0, 1, 2, 3, 4], min_step_tokens=1)

    assert endpoints == [[0, 4]]


def test_blank_lines_inside_code_fence_are_not_split():
    tokenizer = PieceTokenizer({0: "Before\n```python\n", 1: "x = 1\n\n", 2: "y = 2\n```", 3: " after"})

    endpoints = _endpoint_indices(tokenizer, [0, 1, 2, 3], min_step_tokens=1)

    assert endpoints == [[0, 3]]


def test_long_step_uses_sentence_boundary_before_hard_limit():
    tokenizer = PieceTokenizer({0: "word ", 1: "end. "})
    token_ids = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    spans = split_token_ids(tokenizer, token_ids, min_step_tokens=2, max_step_tokens=5)

    assert [(span.start, span.end, span.end_boundary_type) for span in spans] == [
        (0, 4, "sentence_fallback"),
        (4, 8, "forced"),
        (8, 10, "response_end"),
    ]


def test_long_step_prefers_line_boundary_to_forced_split():
    tokenizer = PieceTokenizer({0: "word ", 1: "\n"})
    token_ids = [0, 0, 0, 1, 0, 0, 0]

    spans = split_token_ids(tokenizer, token_ids, min_step_tokens=2, max_step_tokens=4)

    assert [(span.start, span.end, span.end_boundary_type) for span in spans] == [
        (0, 4, "line_fallback"),
        (4, 7, "response_end"),
    ]


def test_long_step_without_safe_text_boundary_is_forced():
    tokenizer = PieceTokenizer({0: "word "})

    spans = split_token_ids(tokenizer, [0] * 10, min_step_tokens=2, max_step_tokens=4)

    assert [(span.start, span.end, span.end_boundary_type) for span in spans] == [
        (0, 4, "forced"),
        (4, 8, "forced"),
        (8, 10, "response_end"),
    ]


def test_splitter_never_retokenizes_generated_ids():
    class NoEncodeTokenizer(PieceTokenizer):
        def __call__(self, text, **kwargs):
            raise AssertionError("step splitting must not re-tokenize generated text")

    tokenizer = NoEncodeTokenizer({0: "First", 1: "\n", 2: "\n", 3: "Second"})

    endpoints = _endpoint_indices(tokenizer, [0, 1, 2, 3], min_step_tokens=1)

    assert endpoints == [[0, 2], [0, 3]]


def test_splitter_resolves_characters_split_across_byte_tokens():
    class BytePieceTokenizer(PieceTokenizer):
        def decode(self, input_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert not skip_special_tokens
            assert not clean_up_tokenization_spaces
            if input_ids == [1]:
                return " �"
            if input_ids == [2]:
                return "�"
            if input_ids == [1, 2]:
                return " 步"
            return super().decode(
                input_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

    tokenizer = BytePieceTokenizer({0: "First\n", 1: "unused", 2: "unused", 3: "骤 2: next"})

    endpoints = _endpoint_indices(tokenizer, [0, 1, 2, 3], min_step_tokens=1)

    assert endpoints == [[0, 0], [0, 3]]


def test_maps_steps_to_original_masked_positions_and_removes_trailing_eos():
    tokenizer = PieceTokenizer({0: "Start.", 1: "\n", 2: "\n", 3: "Next"})
    responses = torch.tensor([[99, 0, 1, 99, 2, 3, 98, 98, 99]])
    response_mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 1, 0]])

    step_end_mask = build_step_end_mask(
        tokenizer,
        responses,
        response_mask,
        min_step_tokens=1,
    )

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 4], [0, 5]]


def test_immediate_eos_is_the_only_step_endpoint():
    tokenizer = PieceTokenizer({0: "unused"})
    responses = torch.tensor([[99, 98, 99]])
    response_mask = torch.tensor([[0, 1, 0]])

    step_end_mask = build_step_end_mask(tokenizer, responses, response_mask)

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 1]]


@pytest.mark.parametrize(
    ("min_step_tokens", "max_step_tokens", "message"),
    [
        (0, 4, "min_step_tokens"),
        (4, 3, "max_step_tokens"),
        (True, 4, "min_step_tokens"),
    ],
)
def test_rejects_invalid_step_lengths(min_step_tokens, max_step_tokens, message):
    tokenizer = PieceTokenizer({0: "unused"})
    with pytest.raises(ValueError, match=message):
        split_token_ids(
            tokenizer,
            [0],
            min_step_tokens=min_step_tokens,
            max_step_tokens=max_step_tokens,
        )


def test_rejects_response_without_any_active_token():
    tokenizer = PieceTokenizer({0: "unused"})
    responses = torch.tensor([[98, 99]])
    response_mask = torch.zeros_like(responses)

    with pytest.raises(ValueError, match="no active response tokens"):
        build_step_end_mask(tokenizer, responses, response_mask)
