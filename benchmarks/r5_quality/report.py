# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed aggregation of the fixed paired R5 quality suite."""

import argparse
import json
from collections import Counter
from pathlib import Path

from run import load_suite
from score import INVALID, _correct, _semantic_answer, _syntax_valid


def load_records(path, expected):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
        raise ValueError("Missing, duplicate or unexpected quality results")
    return {row["id"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_suite(args.suite)
    expected = {c["id"] for c in cases}
    arms = {
        mode: load_records(args.directory / f"{mode}.jsonl", expected)
        for mode in ("baseline", "candidate")
    }
    human_ids = {c["id"] for c in cases if c["task"] == "humaneval"}
    judges = {}
    for mode in arms:
        rows = json.loads((args.directory / f"{mode}-humaneval.json").read_text())[
            "cases"
        ]
        if len(rows) != 164 or {r["id"] for r in rows} != human_ids:
            raise ValueError("Incomplete isolated HumanEval grading")
        judges[mode] = {r["id"]: r for r in rows}
    output = {"tasks": {}, "cases": [], "unapplied_is_not_candidate_coverage": True}
    new_invalid = [
        cid
        for cid in sorted(human_ids)
        if _syntax_valid(arms["baseline"][cid])
        and not _syntax_valid(arms["candidate"][cid])
    ]
    output["new_syntax_invalid_cases"] = new_invalid
    output["syntax_check_origin"] = "existing tq_accuracy_eval/score.py _syntax_valid"
    output["syntax_gate_origin"] = "user/plan: no new invalid output in fixed fixtures"
    for task in ("gsm8k", "mmlu", "humaneval"):
        task_cases = [c for c in cases if c["task"] == task]
        summary = {}
        for mode, rows in arms.items():
            selected = [rows[c["id"]] for c in task_cases]
            summary[mode] = {
                "correct": sum(
                    judges[mode][r["id"]]["passed"]
                    if task == "humaneval"
                    else bool(_correct(r))
                    for r in selected
                ),
                "cases": len(selected),
                "empty": sum(r["empty"] for r in selected),
                "truncated": sum(r["truncated"] for r in selected),
                "answer_parse_invalid": sum(
                    _semantic_answer(r) == INVALID for r in selected
                )
                if task != "humaneval"
                else None,
                "finite_logprobs": all(r["finite_logprobs"] for r in selected),
                "coverage": dict(Counter(r["hook"]["coverage"] for r in selected)),
                "eligible_requests": sum(r["hook"]["calls"] > 0 for r in selected),
                "applied_calls": sum(r["hook"]["applied_calls"] for r in selected),
                "syntax_valid": sum(_syntax_valid(r) for r in selected)
                if task == "humaneval"
                else None,
            }
        summary["score_finite_gate_pass"] = (
            summary["candidate"]["correct"] >= summary["baseline"]["correct"]
            and summary["candidate"]["finite_logprobs"]
            and summary["baseline"]["finite_logprobs"]
            and summary["candidate"]["empty"] <= summary["baseline"]["empty"]
        )
        output["tasks"][task] = summary
    for case in cases:
        cid = case["id"]
        left, right = arms["baseline"][cid], arms["candidate"][cid]
        output["cases"].append(
            {
                "id": cid,
                "task": case["task"],
                "baseline_correct": judges["baseline"][cid]["passed"]
                if cid in human_ids
                else _correct(left),
                "candidate_correct": judges["candidate"][cid]["passed"]
                if cid in human_ids
                else _correct(right),
                "token_equal": left["output_token_ids"] == right["output_token_ids"],
                "candidate_applied": right["hook"]["applied_calls"],
            }
        )
    output["task_score_gate_pass"] = all(
        r["score_finite_gate_pass"] for r in output["tasks"].values()
    )
    output["pass"] = output["task_score_gate_pass"] and not new_invalid
    output["decision"] = (
        "308-case gate passed; long-context gate required"
        if output["pass"]
        else "quality gate failed; stop"
    )
    with args.output.open("x") as result:
        json.dump(output, result, indent=2)
        result.write("\n")
    print(json.dumps(output["tasks"], indent=2))


if __name__ == "__main__":
    main()
