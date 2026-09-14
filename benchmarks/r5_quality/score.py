# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve the existing task scoring rules, with repository import style."""

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import regex as re

INVALID = "<invalid>"


def _load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {row["id"]: row for row in rows}


def _last_integer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{\s*(-?[\d,]+)\s*\}", text)
    numbers = re.findall(r"-?[\d,]+", text)
    values = boxed or numbers
    return values[-1].replace(",", "") if values else INVALID


def _gsm_reference(text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match is None:
        raise ValueError(f"missing GSM8K reference answer: {text!r}")
    return match.group(1).replace(",", "")


def _mmlu_answer(text: str) -> str:
    matches = re.findall(r"(?<![A-Z])[ABCD](?![A-Z])", text.upper())
    return matches[-1] if matches else INVALID


def _human_source(row: dict) -> str:
    text = row["text"].strip()
    fences = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.I)
    if fences:
        text = fences[-1].strip()
    prompt = row["reference"]["prompt"]
    entry_point = row["reference"]["entry_point"]
    if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", text):
        return text
    return prompt + text + "\n"


def _semantic_answer(row: dict) -> str | None:
    if row["task"] == "gsm8k":
        return _last_integer(row["text"])
    if row["task"] == "mmlu":
        return _mmlu_answer(row["text"])
    return None


def _correct(row: dict) -> bool | None:
    if row["task"] == "gsm8k":
        return _last_integer(row["text"]) == _gsm_reference(row["reference"])
    if row["task"] == "mmlu":
        return _mmlu_answer(row["text"]) == row["reference"]
    return None


def _syntax_valid(row: dict) -> bool:
    try:
        ast.parse(_human_source(row))
    except SyntaxError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    a = _load(args.a)
    c = _load(args.c)
    if a.keys() != c.keys():
        raise AssertionError("A/C case sets differ")

    tasks: dict[str, list[str]] = defaultdict(list)
    cases = []
    for case_id in a:
        left, right = a[case_id], c[case_id]
        task = left["task"]
        if task != right["task"]:
            raise AssertionError(case_id)
        token_equal = left["output_token_ids"] == right["output_token_ids"]
        semantic_equal = _semantic_answer(left) == _semantic_answer(right)
        record = {
            "id": case_id,
            "task": task,
            "tokens_equal": token_equal,
            "semantic_answer_equal": semantic_equal,
            "a_correct": _correct(left),
            "c_correct": _correct(right),
        }
        if task == "humaneval":
            record.update(
                a_syntax_valid=_syntax_valid(left),
                c_syntax_valid=_syntax_valid(right),
                generated_source_equal=_human_source(left) == _human_source(right),
            )
        cases.append(record)
        tasks[task].append(case_id)

    summary = {}
    for task, ids in tasks.items():
        selected = [row for row in cases if row["id"] in ids]
        summary[task] = {
            "cases": len(selected),
            "tokens_equal": sum(row["tokens_equal"] for row in selected),
            "semantic_answer_equal": sum(
                row["semantic_answer_equal"] for row in selected
            ),
        }
        if task in {"gsm8k", "mmlu"}:
            summary[task].update(
                a_correct=sum(row["a_correct"] for row in selected),
                c_correct=sum(row["c_correct"] for row in selected),
            )
        else:
            summary[task].update(
                a_syntax_valid=sum(row["a_syntax_valid"] for row in selected),
                c_syntax_valid=sum(row["c_syntax_valid"] for row in selected),
                generated_source_equal=sum(
                    row["generated_source_equal"] for row in selected
                ),
            )
    result = {"summary": summary, "cases": cases}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
