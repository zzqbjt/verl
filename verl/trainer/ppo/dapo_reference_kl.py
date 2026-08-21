# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Build same-group correct-response contexts for the DAPO reference KL loss."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import torch

from verl.utils.model import compute_position_id_with_mask
from verl.utils.tokenizer import normalize_token_ids


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _message_text(content: Any) -> str:
    if hasattr(content, "tolist") and not isinstance(content, str):
        content = content.tolist()
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return _as_text(content)

    text_parts = []
    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(_as_text(part.get("text")))
        elif isinstance(part, dict) and any(key in part for key in ("image", "video", "image_url")):
            raise ValueError("DAPO reference KL currently supports text-only prompts.")
    return "".join(text_parts)


def extract_reference_kl_question(raw_prompt: Any, extra_info: Any, question_key: str) -> str:
    """Extract a question, preferring an explicit field in ``extra_info``."""

    if isinstance(extra_info, dict) and question_key in extra_info:
        question = _as_text(extra_info[question_key]).strip()
        if question:
            return question

    if hasattr(raw_prompt, "tolist") and not isinstance(raw_prompt, str):
        raw_prompt = raw_prompt.tolist()
    if not isinstance(raw_prompt, (list, tuple)):
        question = _as_text(raw_prompt).strip()
        if question:
            return question
        raise ValueError("DAPO reference KL could not extract a non-empty question from raw_prompt.")

    for message in reversed(raw_prompt):
        if isinstance(message, dict) and message.get("role") == "user":
            question = _message_text(message.get("content")).strip()
            if question:
                return question
    raise ValueError("DAPO reference KL raw_prompt must contain a non-empty user message.")


def _strip_student_instruction(question: str, prefix: str, suffix: str) -> str:
    if prefix and question.startswith(prefix):
        question = question[len(prefix) :]
    if suffix and question.endswith(suffix):
        question = question[: -len(suffix)]
    return question.strip()


def _truncate_prompt(token_ids: list[int], max_length: int, truncation: str, row: int) -> list[int]:
    if len(token_ids) <= max_length:
        return token_ids
    if truncation == "error":
        raise ValueError(
            f"DAPO reference teacher prompt at row {row} has {len(token_ids)} tokens, exceeding "
            f"dapo_reference_kl.max_teacher_prompt_length={max_length}."
        )
    if truncation == "left":
        return token_ids[-max_length:]
    return token_ids[:max_length]


def _decode_response(tokenizer, response: torch.Tensor, response_mask: torch.Tensor, row: int) -> str:
    valid_ids = response[response_mask.bool()].tolist()
    solution = tokenizer.decode(valid_ids, skip_special_tokens=True).strip()
    if not solution:
        raise ValueError(f"DAPO reference KL decoded an empty correct response at row {row}.")
    return solution


def _assign_same_group_references(uids, correctness) -> list[int]:
    grouped_rows: dict[Any, list[int]] = defaultdict(list)
    for row, uid in enumerate(uids):
        grouped_rows[uid].append(row)

    reference_rows = [-1] * len(uids)
    for uid, rows in grouped_rows.items():
        correct_rows = [row for row in rows if correctness[row] == 1.0]
        wrong_rows = [row for row in rows if correctness[row] == 0.0]
        if not correct_rows or not wrong_rows:
            raise ValueError(
                "DAPO reference KL expects groups already retained by DAPO dynamic sampling, but "
                f"uid={uid!r} contains {len(correct_rows)} correct and {len(wrong_rows)} incorrect responses."
            )
        for wrong_offset, wrong_row in enumerate(wrong_rows):
            reference_rows[wrong_row] = correct_rows[wrong_offset % len(correct_rows)]
    return reference_rows


def prepare_dapo_reference_kl_inputs(data, tokenizer, config, metric_key: str) -> dict[str, float]:
    """Attach privileged teacher inputs and a loss mask for incorrect rollouts.

    References are verified-correct responses sampled for the same ``uid``.
    Each selected incorrect response is appended unchanged to its privileged
    prompt so teacher and student next-token positions remain aligned.
    """

    required_tensor = ("responses", "response_mask")
    missing_tensor = [key for key in required_tensor if key not in data.batch]
    if missing_tensor:
        raise ValueError(f"DAPO reference KL preparation requires tensor batch keys {missing_tensor}.")

    metric_key = str(metric_key)
    if not metric_key:
        raise ValueError("DAPO reference KL requires a non-empty filter_groups.metric.")
    required_non_tensor = ("raw_prompt", "uid", metric_key)
    missing_non_tensor = [key for key in required_non_tensor if key not in data.non_tensor_batch]
    if missing_non_tensor:
        raise ValueError(f"DAPO reference KL preparation requires non-tensor batch keys {missing_non_tensor}.")
    if "multi_modal_inputs" in data.non_tensor_batch:
        for item in data.non_tensor_batch["multi_modal_inputs"]:
            if item:
                raise ValueError("DAPO reference KL currently supports text-only training examples.")

    responses = data.batch["responses"].detach().cpu().long()
    response_mask = data.batch["response_mask"].detach().cpu().long()
    raw_prompts = data.non_tensor_batch["raw_prompt"]
    extra_infos = data.non_tensor_batch.get("extra_info", [None] * responses.shape[0])
    uids = list(data.non_tensor_batch["uid"])
    correctness = []
    for row, value in enumerate(data.non_tensor_batch[metric_key]):
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"DAPO reference KL correctness value at row {row} must be numeric, got {value!r}."
            ) from exc
        if not math.isfinite(score):
            raise ValueError(f"DAPO reference KL correctness value at row {row} must be finite, got {score}.")
        if score not in (0.0, 1.0):
            raise ValueError(
                f"DAPO reference KL expects binary {metric_key!r} values from DAPO dynamic sampling, "
                f"but row {row} has {score}."
            )
        correctness.append(score)

    reference_rows = _assign_same_group_references(uids, correctness)

    chat_template_kwargs = dict(config.get("teacher_chat_template_kwargs", {}))
    for reserved_key in ("tokenize", "add_generation_prompt", "return_dict", "return_tensors"):
        chat_template_kwargs.pop(reserved_key, None)

    prompt_token_ids: list[list[int] | None] = [None] * responses.shape[0]
    selected_prompt_lengths = []
    for row, reference_row in enumerate(reference_rows):
        if reference_row < 0:
            continue
        question = extract_reference_kl_question(raw_prompts[row], extra_infos[row], str(config.question_key))
        question = _strip_student_instruction(
            question,
            prefix=str(config.get("student_prompt_prefix", "")),
            suffix=str(config.get("student_prompt_suffix", "")),
        )
        if not question:
            raise ValueError(
                f"DAPO reference KL extracted an empty question after removing student instructions at row {row}."
            )
        reference_solution = _decode_response(
            tokenizer,
            responses[reference_row],
            response_mask[reference_row],
            reference_row,
        )
        teacher_text = str(config.teacher_prompt_template).format(
            question=question,
            reference_solution=reference_solution,
        )
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": teacher_text}],
            tokenize=True,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        token_ids = list(normalize_token_ids(encoded))
        token_ids = _truncate_prompt(
            token_ids,
            max_length=int(config.max_teacher_prompt_length),
            truncation=str(config.truncation),
            row=row,
        )
        if not token_ids:
            raise ValueError(f"DAPO reference KL tokenizer produced an empty teacher prompt at row {row}.")
        prompt_token_ids[row] = token_ids
        selected_prompt_lengths.append(len(token_ids))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("DAPO reference KL requires tokenizer.pad_token_id or tokenizer.eos_token_id.")

    batch_size = responses.shape[0]
    prompt_width = max(selected_prompt_lengths, default=1)
    prompt_ids = torch.full((batch_size, prompt_width), int(pad_token_id), dtype=torch.long)
    prompt_mask = torch.zeros((batch_size, prompt_width), dtype=torch.long)
    prompt_lengths = torch.zeros(batch_size, dtype=torch.long)
    selected_rows = torch.zeros(batch_size, dtype=torch.bool)
    for row, token_ids in enumerate(prompt_token_ids):
        if token_ids is None:
            continue
        length = len(token_ids)
        prompt_ids[row, -length:] = torch.tensor(token_ids, dtype=torch.long)
        prompt_mask[row, -length:] = 1
        prompt_lengths[row] = length
        selected_rows[row] = True

    teacher_input_ids = torch.cat((prompt_ids, responses), dim=-1)
    teacher_attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
    teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)
    loss_mask = response_mask * selected_rows.unsqueeze(-1).long()

    data.batch["dapo_reference_teacher_input_ids"] = teacher_input_ids
    data.batch["dapo_reference_teacher_attention_mask"] = teacher_attention_mask
    data.batch["dapo_reference_teacher_position_ids"] = teacher_position_ids
    data.batch["dapo_reference_kl_loss_mask"] = loss_mask
    data.batch["dapo_reference_kl_reference_rows"] = torch.tensor(reference_rows, dtype=torch.long)

    selected_count = int(selected_rows.sum().item())
    selected_tokens = int(loss_mask.sum().item())
    prompt_mean = float(sum(selected_prompt_lengths) / selected_count) if selected_count else 0.0
    prompt_max = float(max(selected_prompt_lengths, default=0))
    return {
        "dapo_reference_kl/incorrect_responses": float(selected_count),
        "dapo_reference_kl/selected_tokens": float(selected_tokens),
        "dapo_reference_kl/teacher_prompt_tokens_mean": prompt_mean,
        "dapo_reference_kl/teacher_prompt_tokens_max": prompt_max,
    }
