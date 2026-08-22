import pytest

pytest.importorskip("math_verify")

from verl.utils.reward_score import default_compute_score, rlvr
from verl.utils.reward_score.rlvr import compute_score


@pytest.mark.parametrize(
    ("solution", "ground_truth"),
    [
        ("Reasoning\nAnswer: 34", "34"),
        ("Reasoning\nAnswer: $\\frac{1}{2}$", "0.5"),
        ("Reasoning\nAnswer: \\sqrt{8}", "2\\sqrt{2}"),
        ("Reasoning\nAnswer: [0,3)", "[0,3)"),
        (
            "Reasoning\nAnswer: \\begin{pmatrix}1&0\\\\0&1\\end{pmatrix}",
            "\\begin{pmatrix}1&0\\\\0&1\\end{pmatrix}",
        ),
        ("Reasoning\nAnswer: 90\\text{ square\nunits}", "90\\text{ square\nunits}"),
        ("Therefore, \\boxed{\\text{odd}}.", "\\text{odd}"),
    ],
)
def test_compute_score_accepts_equivalent_math_formats(solution, ground_truth):
    result = compute_score(solution, ground_truth)
    assert result["score"] == 1.0
    assert result["acc"] is True


def test_compute_score_rejects_incorrect_answer():
    result = compute_score("Reasoning\nAnswer: 7", "8")
    assert result["score"] == -1.0
    assert result["acc"] is False


def test_default_rlvr_dispatch():
    result = default_compute_score(
        data_source="rlvr",
        solution_str="Therefore, \\boxed{2}.",
        ground_truth="2",
    )

    assert result == {"score": 1.0, "acc": True}


def test_score_does_not_stringify_parsed_sympy_objects(monkeypatch):
    class ParsedValue:
        def __str__(self):
            raise AssertionError("parsed values must not be stringified")

    monkeypatch.setattr(rlvr, "parse", lambda *_args, **_kwargs: [ParsedValue()])
    monkeypatch.setattr(rlvr, "verify", lambda *_args, **_kwargs: True)
    rlvr._parse_ground_truth.cache_clear()

    result = compute_score("Answer: 2", "2")

    assert result == {"score": 1.0, "acc": True}


def test_only_bounded_response_tail_is_parsed(monkeypatch):
    parsed_texts = []

    def fake_parse(text, **_kwargs):
        parsed_texts.append(text)
        return [2]

    monkeypatch.setattr(rlvr, "parse", fake_parse)
    monkeypatch.setattr(rlvr, "verify", lambda *_args, **_kwargs: True)
    rlvr._parse_ground_truth.cache_clear()
    response = "discarded-prefix" + "x" * rlvr._MAX_SOLUTION_CHARS

    result = compute_score(response, "2")

    assert result["acc"] is True
    assert parsed_texts[-1] == response[-rlvr._MAX_SOLUTION_CHARS :]
