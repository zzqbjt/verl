#!/usr/bin/env python3
"""Evaluate step-splitting granularity on saved JSON generations using CPU."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.step_split import split_token_ids  # noqa: E402


def _iter_output_texts(payload: Any):
    records = payload
    if isinstance(payload, dict):
        records = payload.get("records", [])
    if not isinstance(records, list):
        return

    for record in records:
        if not isinstance(record, dict):
            continue
        values = None
        for key in ("outputs", "responses", "completions"):
            if key in record:
                values = record[key]
                break
        if isinstance(values, str):
            yield values
        elif isinstance(values, list):
            for value in values:
                if isinstance(value, str):
                    yield value


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(
    response_lengths: list[int],
    step_counts: list[int],
    step_lengths: list[int],
    boundary_counts: Counter[str],
    *,
    min_step_tokens: int,
    max_step_tokens: int,
) -> dict[str, Any]:
    num_responses = len(step_counts)
    num_steps = len(step_lengths)
    num_internal_boundaries = sum(boundary_counts.values())
    return {
        "responses": num_responses,
        "response_tokens": {
            "mean": sum(response_lengths) / num_responses if num_responses else 0.0,
            "p50": _percentile(response_lengths, 0.50),
            "p90": _percentile(response_lengths, 0.90),
            "max": max(response_lengths, default=0),
        },
        "steps_per_response": {
            "mean": sum(step_counts) / num_responses if num_responses else 0.0,
            "p50": _percentile(step_counts, 0.50),
            "p90": _percentile(step_counts, 0.90),
            "max": max(step_counts, default=0),
        },
        "step_tokens": {
            "mean": sum(step_lengths) / num_steps if num_steps else 0.0,
            "p50": _percentile(step_lengths, 0.50),
            "p90": _percentile(step_lengths, 0.90),
            "max": max(step_lengths, default=0),
        },
        "single_step_response_ratio": (
            sum(step_count == 1 for step_count in step_counts) / num_responses if num_responses else 0.0
        ),
        "short_step_ratio": (
            sum(length < min_step_tokens for length in step_lengths) / num_steps if num_steps else 0.0
        ),
        "over_max_step_ratio": (
            sum(length > max_step_tokens for length in step_lengths) / num_steps if num_steps else 0.0
        ),
        "boundary_counts": dict(sorted(boundary_counts.items())),
        "boundary_ratios": {
            kind: count / num_internal_boundaries if num_internal_boundaries else 0.0
            for kind, count in sorted(boundary_counts.items())
        },
    }


def _analyze_file(
    path: Path,
    tokenizer: Any,
    *,
    min_step_tokens: int,
    max_step_tokens: int,
    max_outputs: int,
) -> tuple[dict[str, Any], tuple[list[int], list[int], list[int], Counter[str]]]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    response_lengths: list[int] = []
    step_counts: list[int] = []
    step_lengths: list[int] = []
    boundary_counts: Counter[str] = Counter()

    for output_index, text in enumerate(_iter_output_texts(payload)):
        if max_outputs > 0 and output_index >= max_outputs:
            break
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        if not token_ids:
            continue
        spans = split_token_ids(
            tokenizer,
            token_ids,
            min_step_tokens=min_step_tokens,
            max_step_tokens=max_step_tokens,
        )
        response_lengths.append(len(token_ids))
        step_counts.append(len(spans))
        step_lengths.extend(span.length for span in spans)
        boundary_counts.update(span.end_boundary_type for span in spans if span.end_boundary_type != "response_end")

    summary = _summarize(
        response_lengths,
        step_counts,
        step_lengths,
        boundary_counts,
        min_step_tokens=min_step_tokens,
        max_step_tokens=max_step_tokens,
    )
    return summary, (response_lengths, step_counts, step_lengths, boundary_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--min-step-tokens", type=int, default=96)
    parser.add_argument("--max-step-tokens", type=int, default=512)
    parser.add_argument("--max-outputs-per-file", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    per_file = {}
    all_response_lengths: list[int] = []
    all_step_counts: list[int] = []
    all_step_lengths: list[int] = []
    all_boundary_counts: Counter[str] = Counter()

    for path in args.json_files:
        summary, raw = _analyze_file(
            path,
            tokenizer,
            min_step_tokens=args.min_step_tokens,
            max_step_tokens=args.max_step_tokens,
            max_outputs=args.max_outputs_per_file,
        )
        per_file[str(path)] = summary
        response_lengths, step_counts, step_lengths, boundary_counts = raw
        all_response_lengths.extend(response_lengths)
        all_step_counts.extend(step_counts)
        all_step_lengths.extend(step_lengths)
        all_boundary_counts.update(boundary_counts)

    result = {
        "configuration": {
            "model_path": str(args.model_path),
            "min_step_tokens": args.min_step_tokens,
            "max_step_tokens": args.max_step_tokens,
            "max_outputs_per_file": args.max_outputs_per_file,
        },
        "overall": _summarize(
            all_response_lengths,
            all_step_counts,
            all_step_lengths,
            all_boundary_counts,
            min_step_tokens=args.min_step_tokens,
            max_step_tokens=args.max_step_tokens,
        ),
        "per_file": per_file,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
