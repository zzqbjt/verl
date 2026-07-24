#!/usr/bin/env python3
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
"""Visualize reasoning-step boundaries selected from verl result JSON files.

The report compares the current actor configuration with the original rule that
only recognized ``Step N`` and ``N.`` markers. It uses the requested tokenizer,
so lookahead and fallback distances are measured in model tokens.
"""

import argparse
import html
import json
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from transformers import AutoTokenizer

ORIGINAL_MARKER_PATTERNS = [r"(?i)\bStep\s*\d+\b", r"\b\d+\.\s"]


@dataclass(frozen=True)
class BoundaryCandidate:
    response_position: int
    token_end_idx: int
    matched_text: str | None
    matched_pattern: str | None
    matched_prefix: str | None

    @property
    def is_strong(self) -> bool:
        return self.matched_pattern is not None

    @property
    def is_window_interior(self) -> bool:
        if not self.is_strong or self.matched_prefix is None:
            return False
        prefix_without_formatting = re.sub(r"[\s#*_>`~\-]+", "", self.matched_prefix)
        return bool(prefix_without_formatting)


@dataclass(frozen=True)
class SplitAnalysis:
    token_count: int
    candidates: tuple[BoundaryCandidate, ...]
    selected_positions: tuple[int, ...]
    selected_reasons: dict[int, str]

    @property
    def selected_count(self) -> int:
        return len(self.selected_positions)

    @property
    def has_strong_candidate(self) -> bool:
        return any(candidate.is_strong for candidate in self.candidates)


@dataclass(frozen=True)
class ResponseRecord:
    question: str
    response: str
    row_idx: int
    output_idx: int
    token_count: int
    candidate_count: int
    current_positions: tuple[int, ...]
    original_positions: tuple[int, ...]
    current_strong_count: int
    current_fallback_count: int
    current_window_interior_count: int
    current_marker_labels: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return self.current_positions != self.original_positions

    @property
    def boundary_delta(self) -> int:
        return len(self.current_positions) - len(self.original_positions)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Result JSON containing input/outputs records.")
    parser.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer name or local model directory.")
    parser.add_argument(
        "--actor-config",
        type=Path,
        default=repo_root / "verl/trainer/config/actor/actor.yaml",
        help="Actor YAML providing delimiter and marker defaults.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("step_split_report.html"),
        help="Destination self-contained HTML report.",
    )
    parser.add_argument("--scan-limit", type=int, default=2000, help="Responses to analyze; 0 scans all responses.")
    parser.add_argument("--max-samples", type=int, default=30, help="Responses rendered in detail.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when scan-limit samples the result file.")
    parser.add_argument(
        "--sample-strategy",
        choices=("most-changed", "most-steps", "random"),
        default="most-changed",
        help="How detailed examples are selected from scanned responses.",
    )
    parser.add_argument("--lookahead", type=int, help="Override delimiter_step_marker_lookahead.")
    parser.add_argument("--fallback-min-tokens", type=int, help="Override delimiter_fallback_min_tokens.")
    parser.add_argument("--max-steps-per-response", type=int, help="Override delimiter_max_steps_per_response.")
    parser.add_argument("--step-interval", type=int, help="Override step_interval.")
    parser.add_argument(
        "--marker-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use marker filtering. Training scripts currently enable it.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only load tokenizer files already available locally.",
    )
    return parser.parse_args()


def decode_common_escapes(value: str) -> str:
    return str(value).replace("\\n", "\n").replace("\\t", "\t")


def load_outputs(path: Path) -> list[tuple[str, str, int, int]]:
    with path.open(encoding="utf-8") as result_file:
        data = json.load(result_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")

    responses = []
    for row_idx, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        question = str(row.get("input", ""))
        outputs = row.get("outputs", [])
        if not isinstance(outputs, list):
            continue
        for output_idx, response in enumerate(outputs):
            if isinstance(response, str):
                responses.append((question, response, row_idx, output_idx))
    if not responses:
        raise ValueError(f"No string outputs found in {path}.")
    return responses


def get_delimiter_token_ids(tokenizer, delimiter: str) -> set[int]:
    if not delimiter:
        return set()
    delimiter_ids = set()
    for token_id in set(tokenizer.get_vocab().values()):
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if delimiter in piece:
            delimiter_ids.add(token_id)
    return delimiter_ids


def marker_match_after_delimiter(
    tokenizer,
    response_ids: list[int],
    delimiter_start_idx: int,
    delimiter_end_idx: int,
    delimiter: str,
    lookahead: int,
    compiled_patterns: list[tuple[str, re.Pattern]],
) -> tuple[str | None, str | None, str | None]:
    if lookahead <= 0 or not compiled_patterns:
        return None, None, None
    window_end = min(len(response_ids), delimiter_end_idx + 1 + lookahead)
    window_ids = response_ids[delimiter_start_idx:window_end]
    window_text = tokenizer.decode(
        window_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    delimiter_offset = window_text.find(delimiter) if delimiter else -1
    if delimiter_offset >= 0:
        marker_text = window_text[delimiter_offset + len(delimiter) :]
    else:
        marker_text = tokenizer.decode(
            response_ids[delimiter_end_idx + 1 : window_end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    marker_text = marker_text.lstrip()
    for pattern_text, pattern in compiled_patterns:
        match = pattern.search(marker_text)
        if match is not None:
            return match.group(0), pattern_text, marker_text[: match.start()]
    return None, None, None


def analyze_response(
    tokenizer,
    response: str,
    delimiter: str,
    delimiter_token_ids: set[int],
    delimiter_token_sequence: list[int],
    marker_patterns: list[str],
    lookahead: int,
    fallback_min_tokens: int,
    max_steps_per_response: int,
    step_interval: int,
    marker_filter: bool,
) -> SplitAnalysis:
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    response_positions = list(range(len(response_ids)))
    raw_candidates = [
        (position, idx, idx)
        for idx, (token_id, position) in enumerate(zip(response_ids, response_positions, strict=True))
        if token_id in delimiter_token_ids
    ]
    if delimiter_token_sequence:
        sequence_length = len(delimiter_token_sequence)
        for start_idx in range(len(response_ids) - sequence_length + 1):
            if response_ids[start_idx : start_idx + sequence_length] == delimiter_token_sequence:
                raw_candidates.append(
                    (response_positions[start_idx + sequence_length - 1], start_idx, start_idx + sequence_length - 1)
                )

    ranges_by_position: dict[int, list[tuple[int, int]]] = {}
    for position, start_idx, end_idx in raw_candidates:
        ranges_by_position.setdefault(position, []).append((start_idx, end_idx))

    compiled_patterns = [(pattern, re.compile(pattern)) for pattern in marker_patterns]
    candidates = []
    for position in sorted(ranges_by_position):
        ranges = ranges_by_position[position]
        token_end_idx = max(end_idx for _, end_idx in ranges)
        matched_text = None
        matched_pattern = None
        matched_prefix = None
        if marker_filter:
            for start_idx, end_idx in ranges:
                matched_text, matched_pattern, matched_prefix = marker_match_after_delimiter(
                    tokenizer=tokenizer,
                    response_ids=response_ids,
                    delimiter_start_idx=start_idx,
                    delimiter_end_idx=end_idx,
                    delimiter=delimiter,
                    lookahead=lookahead,
                    compiled_patterns=compiled_patterns,
                )
                if matched_pattern is not None:
                    break
        candidates.append(
            BoundaryCandidate(
                response_position=position,
                token_end_idx=token_end_idx,
                matched_text=matched_text,
                matched_pattern=matched_pattern,
                matched_prefix=matched_prefix,
            )
        )

    if not marker_filter:
        selected_candidates = list(candidates)
        reason_kind = "all-delimiters"
    else:
        strong_candidates = [candidate for candidate in candidates if candidate.is_strong]
        if strong_candidates:
            selected_candidates = strong_candidates
            reason_kind = "strong"
        else:
            selected_candidates = []
            fallback_min_tokens = int(fallback_min_tokens)
            if fallback_min_tokens > 0:
                last_selected_end_idx = -1
                for candidate in candidates:
                    if candidate.token_end_idx - last_selected_end_idx >= fallback_min_tokens:
                        selected_candidates.append(candidate)
                        last_selected_end_idx = candidate.token_end_idx
            reason_kind = "fallback"

    step_interval = max(int(step_interval), 1)
    selected_candidates = [
        candidate
        for step_idx, candidate in enumerate(selected_candidates, start=1)
        if step_idx % step_interval == 0 or step_idx == len(selected_candidates)
    ]
    max_steps = int(max_steps_per_response)
    if max_steps > 0 and len(selected_candidates) > max_steps:
        if max_steps == 1:
            selected_candidates = [selected_candidates[-1]]
        else:
            last_idx = len(selected_candidates) - 1
            selected_candidates = [
                selected_candidates[round(slot * last_idx / (max_steps - 1))] for slot in range(max_steps)
            ]
    selected_positions = tuple(candidate.response_position for candidate in selected_candidates)
    selected_reasons = {}
    for candidate in selected_candidates:
        if reason_kind == "strong":
            location = "window-interior" if candidate.is_window_interior else "paragraph-start"
            selected_reasons[candidate.response_position] = f"strong/{location}: {candidate.matched_text!r}"
        else:
            selected_reasons[candidate.response_position] = reason_kind
    return SplitAnalysis(
        token_count=len(response_ids),
        candidates=tuple(candidates),
        selected_positions=selected_positions,
        selected_reasons=selected_reasons,
    )


def percentile(values: list[int], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * ratio)), len(ordered) - 1)
    return float(ordered[index])


def summarize(records: list[ResponseRecord]) -> dict[str, object]:
    current_counts = [len(record.current_positions) for record in records]
    original_counts = [len(record.original_positions) for record in records]
    marker_counts = Counter(label for record in records for label in record.current_marker_labels)
    return {
        "responses": len(records),
        "responses_changed": sum(record.changed for record in records),
        "responses_with_candidates": sum(record.candidate_count > 0 for record in records),
        "responses_with_current_boundaries": sum(bool(record.current_positions) for record in records),
        "responses_using_fallback": sum(record.current_fallback_count > 0 for record in records),
        "window_interior_boundaries": sum(record.current_window_interior_count for record in records),
        "current_boundaries": sum(current_counts),
        "original_boundaries": sum(original_counts),
        "current_mean": statistics.fmean(current_counts) if current_counts else 0.0,
        "original_mean": statistics.fmean(original_counts) if original_counts else 0.0,
        "current_median": statistics.median(current_counts) if current_counts else 0.0,
        "original_median": statistics.median(original_counts) if original_counts else 0.0,
        "current_p90": percentile(current_counts, 0.9),
        "original_p90": percentile(original_counts, 0.9),
        "current_max": max(current_counts, default=0),
        "original_max": max(original_counts, default=0),
        "top_current_markers": marker_counts.most_common(20),
    }


def normalize_marker_label(matched_text: str | None) -> str:
    if matched_text is None:
        return "128-token fallback"
    label = re.sub(r"\d+", "N", matched_text)
    label = re.sub(r"\s+", " ", label).strip(" #*_\t\r\n")
    return label.casefold() or repr(matched_text)


def histogram_rows(records: list[ResponseRecord]) -> str:
    current = Counter(min(len(record.current_positions), 20) for record in records)
    original = Counter(min(len(record.original_positions), 20) for record in records)
    max_count = max([*current.values(), *original.values(), 1])
    rows = []
    for bucket in sorted(set(current) | set(original)):
        label = "20+" if bucket == 20 else str(bucket)
        current_width = 100 * current[bucket] / max_count
        original_width = 100 * original[bucket] / max_count
        rows.append(
            "<tr>"
            f"<td>{label}</td><td>{current[bucket]}</td>"
            f'<td><div class="bar current" style="width:{current_width:.1f}%"></div></td>'
            f"<td>{original[bucket]}</td>"
            f'<td><div class="bar original" style="width:{original_width:.1f}%"></div></td>'
            "</tr>"
        )
    return "".join(rows)


def marker_rows(records: list[ResponseRecord]) -> str:
    counts = Counter(label for record in records for label in record.current_marker_labels)
    total = max(sum(counts.values()), 1)
    rows = []
    for label, count in counts.most_common(25):
        rows.append(
            f"<tr><td><code>{html.escape(label)}</code></td><td>{count:,}</td><td>{count / total:.1%}</td></tr>"
        )
    return "".join(rows)


def render_segments(tokenizer, response: str, analysis: SplitAnalysis) -> str:
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    candidate_by_position = {candidate.response_position: candidate for candidate in analysis.candidates}
    parts = []
    start_idx = 0
    for segment_idx, position in enumerate(analysis.selected_positions, start=1):
        segment_text = tokenizer.decode(
            response_ids[start_idx : position + 1],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        token_length = position + 1 - start_idx
        reason = analysis.selected_reasons[position]
        candidate = candidate_by_position[position]
        pattern = candidate.matched_pattern or "128-token fallback"
        kind = "strong" if candidate.is_strong else "fallback"
        parts.append(
            f'<div class="segment segment-{segment_idx % 4}">'
            f'<div class="segment-title">Segment {segment_idx} · {token_length} tokens</div>'
            f"<pre>{html.escape(segment_text)}</pre>"
            "</div>"
            f'<div class="boundary {kind}" title="{html.escape(pattern)}">'
            f"BOUNDARY · {html.escape(reason)}</div>"
        )
        start_idx = position + 1

    tail_text = tokenizer.decode(
        response_ids[start_idx:],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    parts.append(
        '<div class="segment tail">'
        f'<div class="segment-title">Tail · {len(response_ids) - start_idx} tokens</div>'
        f"<pre>{html.escape(tail_text)}</pre>"
        "</div>"
    )
    return "".join(parts)


def metric_card(label: str, value: str, note: str = "") -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div>'
        "</div>"
    )


def render_report(
    output_path: Path,
    result_path: Path,
    tokenizer_name: str,
    tokenizer,
    records: list[ResponseRecord],
    samples: list[ResponseRecord],
    current_kwargs: dict,
    original_kwargs: dict,
) -> None:
    summary = summarize(records)
    changed_ratio = summary["responses_changed"] / max(summary["responses"], 1)
    fallback_ratio = summary["responses_using_fallback"] / max(summary["responses"], 1)
    interior_ratio = summary["window_interior_boundaries"] / max(summary["current_boundaries"], 1)
    cards = "".join(
        [
            metric_card("Scanned responses", f"{summary['responses']:,}"),
            metric_card("Changed vs original", f"{changed_ratio:.1%}", f"{summary['responses_changed']:,} responses"),
            metric_card(
                "Current boundaries",
                f"{summary['current_boundaries']:,}",
                f"mean {summary['current_mean']:.2f} · p90 {summary['current_p90']:.0f}",
            ),
            metric_card(
                "Original boundaries",
                f"{summary['original_boundaries']:,}",
                f"mean {summary['original_mean']:.2f} · p90 {summary['original_p90']:.0f}",
            ),
            metric_card(
                "Fallback responses",
                f"{fallback_ratio:.1%}",
                f"{summary['responses_using_fallback']:,} responses",
            ),
            metric_card(
                "Window-interior boundaries",
                f"{interior_ratio:.1%}",
                f"{summary['window_interior_boundaries']:,} boundaries",
            ),
            metric_card(
                "Current max steps",
                f"{summary['current_max']}",
                f"original max {summary['original_max']}",
            ),
        ]
    )

    sample_html = []
    for sample_idx, record in enumerate(samples, start=1):
        current = analyze_response(response=record.response, **current_kwargs)
        original = analyze_response(response=record.response, **original_kwargs)
        sample_html.append(
            '<details class="sample" open>'
            f"<summary>#{sample_idx} · row {record.row_idx}, output {record.output_idx} · "
            f"{record.token_count} tokens · original {len(record.original_positions)} → "
            f"current {len(record.current_positions)}</summary>"
            f'<div class="question"><b>Prompt:</b> {html.escape(record.question)}</div>'
            '<div class="columns">'
            "<section><h3>Current rule</h3>"
            f"<div class=small>{len(current.candidates)} double-newline candidates</div>"
            f"{render_segments(tokenizer, record.response, current)}</section>"
            "<section><h3>Original Step/number rule</h3>"
            f"<div class=small>{len(original.candidates)} double-newline candidates</div>"
            f"{render_segments(tokenizer, record.response, original)}</section>"
            "</div></details>"
        )

    configuration = {
        "results": str(result_path),
        "tokenizer": tokenizer_name,
        "lookahead": current_kwargs["lookahead"],
        "fallback_min_tokens": current_kwargs["fallback_min_tokens"],
        "max_steps_per_response": current_kwargs["max_steps_per_response"],
        "step_interval": current_kwargs["step_interval"],
        "marker_filter": current_kwargs["marker_filter"],
        "current_pattern_count": len(current_kwargs["marker_patterns"]),
    }
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>verl step-splitting report</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#687086; --line:#d9dfeb; --paper:#f5f7fb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14px/1.5 ui-sans-serif,system-ui,sans-serif; color:var(--ink); background:var(--paper); }}
main {{ width:min(1600px,96vw); margin:28px auto 80px; }}
h1 {{ margin-bottom:4px; }} h2 {{ margin-top:32px; }} h3 {{ margin:0 0 4px; }}
.subtitle,.small {{ color:var(--muted); }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:22px 0; }}
.metric,.panel,.sample {{
  background:white; border:1px solid var(--line); border-radius:10px; box-shadow:0 2px 8px #1720330a;
}}
.metric {{ padding:14px; }} .metric-label {{ color:var(--muted); }} .metric-value {{ font-size:25px; font-weight:700; }}
.metric-note {{ color:var(--muted); min-height:20px; }} .panel {{ padding:16px; margin:14px 0; overflow:auto; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:5px 8px; text-align:left; border-bottom:1px solid #edf0f5; }}
.bar {{ min-width:2px; height:10px; border-radius:5px; }}
.bar.current {{ background:#5b6ee1; }} .bar.original {{ background:#9ba5bb; }}
.sample {{ margin:14px 0; overflow:hidden; }}
summary {{ cursor:pointer; padding:13px 16px; font-weight:700; background:#fdfdff; }}
.question {{ padding:12px 16px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
.columns {{ display:grid; grid-template-columns:1fr 1fr; gap:0; }} .columns section {{ padding:14px; min-width:0; }}
.columns section+section {{ border-left:1px solid var(--line); }}
.segment {{ margin-top:10px; border:1px solid var(--line); border-radius:7px; overflow:hidden; background:#fff; }}
.segment-title {{ padding:4px 8px; font-size:12px; font-weight:700; color:#4b5570; background:#eef2ff; }}
.segment-1 .segment-title {{ background:#e8f7ef; }} .segment-2 .segment-title {{ background:#fff3dd; }}
.segment-3 .segment-title {{ background:#e8f2ff; }} .segment-0 .segment-title {{ background:#f5eafe; }}
.tail .segment-title {{ background:#eceff4; }}
pre {{ margin:0; padding:9px; white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.45 ui-monospace,monospace; }}
.boundary {{
  width:max-content; max-width:100%; margin:5px auto; padding:3px 8px;
  border-radius:999px; font-size:11px; font-weight:800;
}}
.boundary.strong {{ color:#075d3c; background:#d7f7e8; }} .boundary.fallback {{ color:#8a4b00; background:#ffe8bd; }}
code {{ white-space:pre-wrap; }}
@media (max-width:900px) {{
  .columns {{ grid-template-columns:1fr; }}
  .columns section+section {{ border-left:0; border-top:1px solid var(--line); }}
}}
</style>
</head>
<body><main>
<h1>verl reasoning-step split report</h1>
<div class="subtitle">Current actor rule compared with the original Step/number-only rule.</div>
<div class="metrics">{cards}</div>
<div class="panel"><b>Configuration</b>
<pre>{html.escape(json.dumps(configuration, indent=2, ensure_ascii=False))}</pre></div>
<h2>Selected-boundary count distribution</h2>
<div class="panel"><table><thead><tr>
<th>Steps</th><th>Current responses</th><th>Current</th>
<th>Original responses</th><th>Original</th></tr></thead>
<tbody>{histogram_rows(records)}</tbody></table></div>
<h2>Top matched markers under the current rule</h2>
<div class="panel"><table><thead><tr><th>Marker</th><th>Boundaries</th><th>Share</th></tr></thead>
<tbody>{marker_rows(records)}</tbody></table></div>
<h2>Detailed examples</h2>
{"".join(sample_html)}
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    with args.actor_config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    delimiter = decode_common_escapes(config.get("delimiter", "\n\n"))
    lookahead = args.lookahead if args.lookahead is not None else int(config.get("delimiter_step_marker_lookahead", 10))
    fallback_min_tokens = (
        args.fallback_min_tokens
        if args.fallback_min_tokens is not None
        else int(config.get("delimiter_fallback_min_tokens", 0))
    )
    max_steps_per_response = (
        args.max_steps_per_response
        if args.max_steps_per_response is not None
        else int(config.get("delimiter_max_steps_per_response", 0))
    )
    step_interval = args.step_interval if args.step_interval is not None else int(config.get("step_interval", 1))
    current_patterns = [decode_common_escapes(pattern) for pattern in config["delimiter_step_marker_patterns"]]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)
    delimiter_token_ids = get_delimiter_token_ids(tokenizer, delimiter)
    delimiter_token_sequence = tokenizer.encode(delimiter, add_special_tokens=False) if delimiter else []
    all_outputs = load_outputs(args.results)
    rng = random.Random(args.seed)
    if args.scan_limit > 0 and len(all_outputs) > args.scan_limit:
        scanned_outputs = rng.sample(all_outputs, args.scan_limit)
    else:
        scanned_outputs = all_outputs

    shared_kwargs = {
        "tokenizer": tokenizer,
        "delimiter": delimiter,
        "delimiter_token_ids": delimiter_token_ids,
        "delimiter_token_sequence": delimiter_token_sequence,
        "lookahead": lookahead,
        "step_interval": step_interval,
        "marker_filter": args.marker_filter,
    }
    current_kwargs = {
        **shared_kwargs,
        "marker_patterns": current_patterns,
        "fallback_min_tokens": fallback_min_tokens,
        "max_steps_per_response": max_steps_per_response,
    }
    original_kwargs = {
        **shared_kwargs,
        "marker_patterns": ORIGINAL_MARKER_PATTERNS,
        "fallback_min_tokens": 10**18,
        "max_steps_per_response": 0,
    }

    records = []
    for question, response, row_idx, output_idx in scanned_outputs:
        current = analyze_response(response=response, **current_kwargs)
        original = analyze_response(response=response, **original_kwargs)
        current_strong_count = sum(
            current.selected_reasons[position].startswith("strong") for position in current.selected_positions
        )
        candidate_by_position = {candidate.response_position: candidate for candidate in current.candidates}
        current_window_interior_count = sum(
            candidate_by_position[position].is_window_interior for position in current.selected_positions
        )
        current_marker_labels = tuple(
            normalize_marker_label(candidate_by_position[position].matched_text)
            for position in current.selected_positions
        )
        records.append(
            ResponseRecord(
                question=question,
                response=response,
                row_idx=row_idx,
                output_idx=output_idx,
                token_count=current.token_count,
                candidate_count=len(current.candidates),
                current_positions=current.selected_positions,
                original_positions=original.selected_positions,
                current_strong_count=current_strong_count,
                current_fallback_count=len(current.selected_positions) - current_strong_count,
                current_window_interior_count=current_window_interior_count,
                current_marker_labels=current_marker_labels,
            )
        )

    if args.sample_strategy == "most-changed":
        ranked = sorted(
            records,
            key=lambda record: (abs(record.boundary_delta), len(record.current_positions), record.token_count),
            reverse=True,
        )
    elif args.sample_strategy == "most-steps":
        ranked = sorted(records, key=lambda record: (len(record.current_positions), record.token_count), reverse=True)
    else:
        ranked = list(records)
        rng.shuffle(ranked)
    samples = ranked[: max(args.max_samples, 0)]

    render_report(
        output_path=args.output,
        result_path=args.results,
        tokenizer_name=args.tokenizer,
        tokenizer=tokenizer,
        records=records,
        samples=samples,
        current_kwargs=current_kwargs,
        original_kwargs=original_kwargs,
    )
    summary = summarize(records)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
