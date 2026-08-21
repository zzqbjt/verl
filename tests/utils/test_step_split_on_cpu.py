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

from verl.utils.step_split import build_step_end_mask, build_step_start_mask


class PieceTokenizer:
    """Small reversible tokenizer whose IDs each cover one configured piece."""

    def __init__(self, pieces, *, eos_token_id=98, pad_token_id=99):
        self.pieces = dict(pieces)
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self._piece_ids = sorted(self.pieces, key=lambda token_id: len(self.pieces[token_id]), reverse=True)

    def decode(self, input_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(self.pieces[token_id] for token_id in input_ids)

    def __call__(self, text, *, add_special_tokens, return_attention_mask, return_offsets_mapping):
        assert not add_special_tokens
        assert not return_attention_mask
        assert return_offsets_mapping
        input_ids = []
        offsets = []
        char_start = 0
        while char_start < len(text):
            for token_id in self._piece_ids:
                piece = self.pieces[token_id]
                if text.startswith(piece, char_start):
                    input_ids.append(token_id)
                    offsets.append((char_start, char_start + len(piece)))
                    char_start += len(piece)
                    break
            else:
                raise AssertionError(f"No configured token piece starts at character {char_start}: {text!r}")
        return {"input_ids": input_ids, "offset_mapping": offsets}


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


@pytest.fixture
def tokenizer():
    return PieceTokenizer(
        {
            0: "Intro",
            1: "!\n\n",
            2: "### ",
            3: "Step ",
            4: "1",
            5: ": do one",
            6: ".\n\n",
            7: "步骤",
            8: "：",
            9: "2",
            10: "：do two",
        }
    )


def test_numeric_markers_merge_step_one_preamble_and_strip_trailing_eos(tokenizer):
    responses = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 98, 99]])
    response_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    step_end_mask = build_step_end_mask(tokenizer, responses, response_mask)

    expected = torch.zeros_like(responses, dtype=torch.bool)
    expected[0, [6, 10]] = True
    torch.testing.assert_close(step_end_mask, expected)


def test_separate_preamble_keeps_split_before_step_one(tokenizer):
    responses = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    response_mask = torch.ones_like(responses)

    step_end_mask = build_step_end_mask(
        tokenizer,
        responses,
        response_mask,
        separate_preamble=True,
    )

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 1], [0, 6], [0, 10]]


def test_requires_double_newline_to_be_covered_by_one_token_and_numeric_marker():
    tokenizer = PieceTokenizer(
        {
            0: "First",
            1: "\n",
            2: "\n ",
            3: "Step ",
            4: "2",
            5: " body",
            6: ".\n\n",
            7: "Next, continue",
            8: " final",
            9: "First\n\n ",
        }
    )
    responses = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
    text = tokenizer.decode(
        responses[0].tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assert (
        tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )["input_ids"]
        != responses[0].tolist()
    )

    step_end_mask = build_step_end_mask(tokenizer, responses, torch.ones_like(responses))

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 8]]


def test_splitter_does_not_retokenize_generated_ids():
    class NoEncodeTokenizer(PieceTokenizer):
        def __call__(self, text, **kwargs):
            raise AssertionError("step splitting must not re-tokenize generated text")

    tokenizer = NoEncodeTokenizer(
        {
            0: "First.\n\n",
            1: "Step ",
            2: "2",
            3: " body",
        }
    )
    responses = torch.tensor([[0, 1, 2, 3]])

    step_end_mask = build_step_end_mask(tokenizer, responses, torch.ones_like(responses))

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 0], [0, 3]]


@pytest.mark.parametrize(
    "delimiter_piece",
    [":\n\n", ",\n\n", ".\n\n", " arbitrary token prefix!?\n\n"],
)
def test_any_token_containing_double_newline_can_be_a_delimiter(delimiter_piece):
    tokenizer = PieceTokenizer(
        {
            0: "First",
            1: delimiter_piece,
            2: "**Step ",
            3: "2",
            4: "** body",
        }
    )
    responses = torch.tensor([[0, 1, 2, 3, 4]])

    step_end_mask = build_step_end_mask(tokenizer, responses, torch.ones_like(responses))

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 1], [0, 4]]


@pytest.mark.parametrize(
    "marker_piece",
    [
        "**Step 3**: body",
        "### **Step 3**: body",
        "prefix before **Step 3**: body",
        "3. ordinary step",
        "3) ordinary step",
        "(3) ordinary step",
        "3、普通步骤",
        "### **3. ordinary step**",
        "prefix before **3. ordinary step**",
        "B. alphabetic step",
        "二、中文步骤",
    ],
)
def test_supported_step_marker_forms_are_found_inside_lookahead(marker_piece):
    tokenizer = PieceTokenizer({0: "First.\n\n", 1: marker_piece})
    responses = torch.tensor([[0, 1]])

    step_end_mask = build_step_end_mask(tokenizer, responses, torch.ones_like(responses))

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 0], [0, 1]]


@pytest.mark.parametrize(
    ("first_marker", "second_marker"),
    [
        ("**1. first**", "2) second"),
        ("A. first", "B. second"),
        ("一、第一步", "二、第二步"),
    ],
)
def test_ordinary_first_marker_merges_preamble(first_marker, second_marker):
    tokenizer = PieceTokenizer(
        {
            0: "Intro.\n\n",
            1: first_marker,
            2: ".\n\n",
            3: second_marker,
        }
    )
    responses = torch.tensor([[0, 1, 2, 3]])

    step_end_mask = build_step_end_mask(tokenizer, responses, torch.ones_like(responses))

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 2], [0, 3]]


def test_adjacent_whitespace_delimiters_collapse_to_the_one_closest_to_marker():
    tokenizer = PieceTokenizer(
        {
            0: "Body",
            1: ".\n\n",
            2: " \n\n",
            3: "**Step ",
            4: "2",
            5: "** next",
        }
    )
    responses = torch.tensor([[0, 1, 2, 3, 4, 5]])

    step_end_mask = build_step_end_mask(tokenizer, responses, torch.ones_like(responses))

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 2], [0, 5]]


def test_adjacent_delimiters_before_step_one_still_merge_preamble():
    tokenizer = PieceTokenizer(
        {
            0: "Intro",
            1: ".\n\n",
            2: " \n\n",
            3: "**Step ",
            4: "1",
            5: "** first",
        }
    )
    responses = torch.tensor([[0, 1, 2, 3, 4, 5]])

    step_end_mask = build_step_end_mask(tokenizer, responses, torch.ones_like(responses))

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 5]]


def test_non_eos_whitespace_only_response_is_one_step():
    tokenizer = PieceTokenizer({0: "\n\n", 1: " \n"})
    responses = torch.tensor([[0, 1]])

    step_end_mask = build_step_end_mask(tokenizer, responses, torch.ones_like(responses))

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 1]]


def test_marker_must_fall_inside_token_lookahead():
    tokenizer = PieceTokenizer(
        {
            0: "First.\n\n",
            1: "filler-a ",
            2: "filler-b ",
            3: "Step ",
            4: "2",
            5: " body",
        }
    )
    responses = torch.tensor([[0, 1, 2, 3, 4, 5]])
    response_mask = torch.ones_like(responses)

    short_mask = build_step_end_mask(tokenizer, responses, response_mask, lookahead_tokens=2)
    long_mask = build_step_end_mask(tokenizer, responses, response_mask, lookahead_tokens=4)

    assert short_mask.nonzero(as_tuple=False).tolist() == [[0, 5]]
    assert long_mask.nonzero(as_tuple=False).tolist() == [[0, 0], [0, 5]]


def test_maps_steps_back_to_original_masked_positions_and_removes_all_trailing_eos():
    tokenizer = PieceTokenizer({0: "Start.\n\n", 1: "Step ", 2: "2", 3: " body"})
    responses = torch.tensor([[99, 0, 1, 99, 2, 3, 98, 98, 99]])
    response_mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 1, 0]])

    step_end_mask = build_step_end_mask(tokenizer, responses, response_mask)

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 1], [0, 5]]


def test_noncanonical_generated_tokenization_keeps_original_step_positions():
    tokenizer = PieceTokenizer(
        {
            0: "Preamble.\n\n",
            1: "Step ",
            2: "1",
            3: ": Ass",
            4: "ist",
            5: " first.\n\n",
            6: "Step ",
            7: "2",
            8: ": done",
            9: ": Assist",
        }
    )
    original_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    text = tokenizer.decode(
        original_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    retokenized_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )["input_ids"]
    assert retokenized_ids != original_ids

    responses = torch.tensor([original_ids])
    step_end_mask = build_step_end_mask(tokenizer, responses, torch.ones_like(responses))

    # The Step-1 preamble is merged.  The Step-2 delimiter and final endpoint
    # remain aligned to the original sampled actions despite the BPE merge.
    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 5], [0, 8]]


@pytest.mark.parametrize("lookahead_tokens", [0, -1])
def test_lookahead_must_be_positive(tokenizer, lookahead_tokens):
    responses = torch.tensor([[0]])
    with pytest.raises(ValueError, match="lookahead_tokens must be positive"):
        build_step_end_mask(
            tokenizer,
            responses,
            torch.ones_like(responses),
            lookahead_tokens=lookahead_tokens,
        )


def test_immediate_eos_is_the_only_step_endpoint():
    tokenizer = PieceTokenizer({0: "unused"})
    responses = torch.tensor([[99, 98, 99]])
    response_mask = torch.tensor([[0, 1, 0]])

    step_end_mask = build_step_end_mask(tokenizer, responses, response_mask)

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 1]]


def test_repeated_active_eos_uses_the_first_termination_action():
    tokenizer = PieceTokenizer({0: "unused"})
    responses = torch.tensor([[98, 98, 99]])
    response_mask = torch.tensor([[1, 1, 0]])

    step_end_mask = build_step_end_mask(tokenizer, responses, response_mask)

    assert step_end_mask.nonzero(as_tuple=False).tolist() == [[0, 0]]


def test_rejects_response_without_any_active_token():
    tokenizer = PieceTokenizer({0: "unused"})
    responses = torch.tensor([[98, 99]])
    response_mask = torch.zeros_like(responses)

    with pytest.raises(ValueError, match="no active response tokens"):
        build_step_end_mask(tokenizer, responses, response_mask)
