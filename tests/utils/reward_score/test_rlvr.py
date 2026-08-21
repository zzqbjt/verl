from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("math_verify")

from verl.utils.reward_score.rlvr import compute_score, extract_answer_candidate, extract_last_boxed


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


def test_extract_answer_candidate_uses_last_nonempty_answer_line():
    solution = "Answer: 3\nMore work\nFinal Answer: $\\frac{7}{2}$"
    assert extract_answer_candidate(solution) == "\\frac{7}{2}"


def test_extract_last_boxed_handles_nested_braces():
    solution = "Earlier \\boxed{1}. Finally \\boxed{\\frac{1}{\\sqrt{2}}}."
    assert extract_last_boxed(solution) == "\\boxed{\\frac{1}{\\sqrt{2}}}"


def test_compute_score_works_in_threaded_dapo_reward_executor():
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(compute_score, "Therefore, \\boxed{2}.", "2").result()

    assert result["score"] == 1.0
    assert result["acc"] is True
