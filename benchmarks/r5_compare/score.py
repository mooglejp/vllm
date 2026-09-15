# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score cross-runtime quality and 32K retention records."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

import regex as re


def _load(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {row["case_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate case IDs in {path}")
    return result


def _last_integer(text: str) -> str | None:
    values = re.findall(r"\boxed\{\s*(-?[\d,]+)\s*\}|(-?[\d,]+)", text)
    flattened = [next(value for value in pair if value) for pair in values]
    return flattened[-1].replace(",", "") if flattened else None


def _mmlu(text: str) -> str | None:
    values = re.findall(r"(?<![A-Z])[ABCD](?![A-Z])", text.upper())
    return values[-1] if values else None


def _human_source(row: dict[str, Any]) -> str:
    text = row["text"].strip()
    fences = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.I)
    if fences:
        text = fences[-1].strip()
    reference = row["reference"]
    if re.search(rf"\bdef\s+{re.escape(reference['entry_point'])}\s*\(", text):
        return text
    return reference["prompt"] + text + "\n"


def _syntax(row: dict[str, Any]) -> bool:
    try:
        ast.parse(_human_source(row))
    except SyntaxError:
        return False
    return True


def _correct(row: dict[str, Any]) -> bool | None:
    if row["task"] == "gsm8k":
        reference = re.search(r"####\s*(-?[\d,]+)", row["reference"])
        return reference is not None and _last_integer(row["text"]) == reference.group(
            1
        ).replace(",", "")
    if row["task"] == "mmlu":
        return _mmlu(row["text"]) == row["reference"]
    return None


def score_quality(suite: Path, vllm: Path, llama: Path) -> dict[str, Any]:
    cases = {
        row["id"]: row
        for row in (json.loads(line) for line in suite.read_text().splitlines())
    }
    left, right = _load(vllm), _load(llama)
    if set(cases) != set(left) or set(cases) != set(right):
        raise ValueError("quality case sets differ")
    paired = []
    for case_id, case in cases.items():
        a, b = left[case_id], right[case_id]
        row = {
            "id": case_id,
            "task": case["task"],
            "vllm": {
                "correct": _correct({**case, **a}),
                "syntax_valid": _syntax({**case, **a})
                if case["task"] == "humaneval"
                else None,
                "finish_reason": a.get("finish_reason"),
                "output_tokens": a.get("output_tokens"),
                "error": a.get("request_error") or a.get("stream_error"),
            },
            "llama": {
                "correct": _correct({**case, **b}),
                "syntax_valid": _syntax({**case, **b})
                if case["task"] == "humaneval"
                else None,
                "finish_reason": b.get("finish_reason"),
                "output_tokens": b.get("output_tokens"),
                "error": b.get("request_error") or b.get("stream_error"),
            },
        }
        row["category"] = (
            "both_correct"
            if row["vllm"]["correct"] and row["llama"]["correct"]
            else "vllm_only"
            if row["vllm"]["correct"]
            else "llama_only"
            if row["llama"]["correct"]
            else "both_incorrect"
        )
        paired.append(row)
    by_task = {}
    for task in ("gsm8k", "mmlu", "humaneval"):
        rows = [row for row in paired if row["task"] == task]
        by_task[task] = {
            "cases": len(rows),
            "categories": dict(Counter(row["category"] for row in rows)),
            "vllm_correct": sum(row["vllm"]["correct"] is True for row in rows),
            "llama_correct": sum(row["llama"]["correct"] is True for row in rows),
            "vllm_truncated": sum(
                row["vllm"]["finish_reason"] == "length" for row in rows
            ),
            "llama_truncated": sum(
                row["llama"]["finish_reason"] == "length" for row in rows
            ),
            "vllm_syntax_valid": sum(
                row["vllm"]["syntax_valid"] is True for row in rows
            ),
            "llama_syntax_valid": sum(
                row["llama"]["syntax_valid"] is True for row in rows
            ),
        }
    return {"tasks": by_task, "cases": paired, "comparison_only": True}


def score_retention(suite: Path, vllm: Path, llama: Path) -> dict[str, Any]:
    cases = {
        row["id"]: row
        for row in (json.loads(line) for line in suite.read_text().splitlines())
    }
    left, right = _load(vllm), _load(llama)
    paired = []
    for case_id, case in cases.items():
        a, b = left[case_id], right[case_id]
        vllm_exact = a.get("text", "").strip() == case["answer"]
        llama_exact = b.get("text", "").strip() == case["answer"]
        paired.append(
            {
                "id": case_id,
                "position": case["position"],
                "category": (
                    "both_correct"
                    if vllm_exact and llama_exact
                    else "vllm_only"
                    if vllm_exact
                    else "llama_only"
                    if llama_exact
                    else "both_incorrect"
                ),
                "vllm_exact": vllm_exact,
                "llama_exact": llama_exact,
                "vllm_finish_reason": a.get("finish_reason"),
                "llama_finish_reason": b.get("finish_reason"),
            }
        )
    return {
        "cases": len(paired),
        "categories": dict(Counter(row["category"] for row in paired)),
        "vllm_correct": sum(row["vllm_exact"] for row in paired),
        "llama_correct": sum(row["llama_exact"] for row in paired),
        "by_position": {
            position: {
                "vllm_correct": sum(
                    row["vllm_exact"] for row in paired if row["position"] == position
                ),
                "llama_correct": sum(
                    row["llama_exact"] for row in paired if row["position"] == position
                ),
            }
            for position in ("early", "middle", "late")
        },
        "paired": paired,
        "comparison_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("quality", "retention"), required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        score_quality(args.suite, args.vllm, args.llama)
        if args.kind == "quality"
        else score_retention(args.suite, args.vllm, args.llama)
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
