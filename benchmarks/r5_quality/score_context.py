# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score the additional context-qualified R5 quality evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_context import load_suite
from score import INVALID, _correct, _semantic_answer, _syntax_valid


def _records(path: Path, expected: set[str]) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != len(expected) or {row["id"] for row in rows} != expected:
        raise ValueError(f"missing, duplicate or unexpected records in {path}")
    return {row["id"]: row for row in rows}


def _human_judges(path: Path, expected: set[str]) -> dict[str, dict]:
    data = json.loads(path.read_text())
    rows = data["cases"]
    if len(rows) != len(expected) or {row["id"] for row in rows} != expected:
        raise ValueError(f"incomplete HumanEval grading in {path}")
    return {row["id"]: row for row in rows}


def _format_invalid(row: dict) -> bool:
    if row["task"] == "humaneval":
        return not _syntax_valid(row)
    return _semantic_answer(row) == INVALID


def _markdown_violation(row: dict) -> bool:
    return row["task"] == "humaneval" and "```" in row["text"]


def _case_summary(
    case: dict,
    baseline: dict,
    candidate: dict,
    judges: dict[str, dict[str, dict]],
) -> dict:
    cid = case["id"]
    result = {
        "id": cid,
        "task": case["task"],
        "baseline_finish_reason": baseline["finish_reason"],
        "candidate_finish_reason": candidate["finish_reason"],
        "baseline_truncated": baseline["truncated"],
        "candidate_truncated": candidate["truncated"],
        "baseline_empty": baseline["empty"],
        "candidate_empty": candidate["empty"],
        "baseline_finite_logprobs": baseline["finite_logprobs"],
        "candidate_finite_logprobs": candidate["finite_logprobs"],
        "baseline_correct": (
            judges["baseline"][cid]["passed"]
            if case["task"] == "humaneval"
            else bool(_correct(baseline))
        ),
        "candidate_correct": (
            judges["candidate"][cid]["passed"]
            if case["task"] == "humaneval"
            else bool(_correct(candidate))
        ),
        "token_equal": baseline["output_token_ids"] == candidate["output_token_ids"],
        "prompt_equal": baseline["prompt_token_ids"] == candidate["prompt_token_ids"],
        "max_tokens_equal": (
            baseline["request_max_tokens"]
            == candidate["request_max_tokens"]
            == case["max_tokens"]
        ),
        "baseline_applied_calls": baseline["hook"]["applied_calls"],
        "candidate_applied_calls": candidate["hook"]["applied_calls"],
        "candidate_applied_overlap_calls": candidate["hook"].get(
            "applied_overlap_calls", 0
        ),
        "same_target_shapes": baseline["hook"]["shapes"] == candidate["hook"]["shapes"],
    }
    result["both_completed"] = (
        baseline["finish_reason"] == candidate["finish_reason"] == "stop"
    )
    if result["both_completed"]:
        result["baseline_format_invalid"] = _format_invalid(baseline)
        result["candidate_format_invalid"] = _format_invalid(candidate)
        result["baseline_markdown_violation"] = _markdown_violation(baseline)
        result["candidate_markdown_violation"] = _markdown_violation(candidate)
    else:
        result["baseline_format_invalid"] = None
        result["candidate_format_invalid"] = None
        result["baseline_markdown_violation"] = None
        result["candidate_markdown_violation"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases, suite_manifest = load_suite(args.suite, args.manifest)
    expected = {case["id"] for case in cases}
    arms = {
        mode: _records(args.directory / f"{mode}.jsonl", expected)
        for mode in ("baseline", "candidate")
    }
    human_ids = {case["id"] for case in cases if case["task"] == "humaneval"}
    judges = {
        mode: _human_judges(args.directory / f"{mode}-humaneval.json", human_ids)
        for mode in ("baseline", "candidate")
    }
    case_rows = [
        _case_summary(
            case, arms["baseline"][case["id"]], arms["candidate"][case["id"]], judges
        )
        for case in cases
    ]
    output = {
        "suite_id": suite_manifest["suite_id"],
        "suite_sha256": suite_manifest["suite_sha256"],
        "rules": {
            "correct_per_task_noninferior": True,
            "truncated_per_task_noninferior": True,
            "completed_pair_format_noninferior": True,
            "no_new_error_empty_nonfinite": True,
            "candidate_coverage_all_308_and_problem_overlap": True,
            "greedy_token_mismatch_is_not_a_failure": True,
            "historical_quality_stop_unchanged": True,
        },
        "tasks": {},
        "coverage": {},
        "cases": case_rows,
    }
    for task in ("gsm8k", "mmlu", "humaneval"):
        selected = [row for row in case_rows if row["task"] == task]
        baseline_rows = [arms["baseline"][row["id"]] for row in selected]
        candidate_rows = [arms["candidate"][row["id"]] for row in selected]
        completed = [row for row in selected if row["both_completed"]]
        task_result = {
            "cases": len(selected),
            "baseline_correct": sum(row["baseline_correct"] for row in selected),
            "candidate_correct": sum(row["candidate_correct"] for row in selected),
            "baseline_truncated": sum(row["baseline_truncated"] for row in selected),
            "candidate_truncated": sum(row["candidate_truncated"] for row in selected),
            "baseline_empty": sum(row["baseline_empty"] for row in selected),
            "candidate_empty": sum(row["candidate_empty"] for row in selected),
            "baseline_finite_logprobs": all(
                row["finite_logprobs"] for row in baseline_rows
            ),
            "candidate_finite_logprobs": all(
                row["finite_logprobs"] for row in candidate_rows
            ),
            "both_completed": len(completed),
            "completed_format_invalid": {
                "baseline": sum(row["baseline_format_invalid"] for row in completed),
                "candidate": sum(row["candidate_format_invalid"] for row in completed),
            },
            "completed_markdown_violations": {
                "baseline": sum(
                    row["baseline_markdown_violation"] for row in completed
                ),
                "candidate": sum(
                    row["candidate_markdown_violation"] for row in completed
                ),
            },
            "new_completed_format_invalid_ids": [
                row["id"]
                for row in completed
                if row["candidate_format_invalid"]
                and not row["baseline_format_invalid"]
            ],
            "resolved_completed_format_invalid_ids": [
                row["id"]
                for row in completed
                if row["baseline_format_invalid"]
                and not row["candidate_format_invalid"]
            ],
            "new_markdown_violation_ids": [
                row["id"]
                for row in completed
                if row["candidate_markdown_violation"]
                and not row["baseline_markdown_violation"]
            ],
        }
        task_result["score_gate_pass"] = (
            task_result["candidate_correct"] >= task_result["baseline_correct"]
            and task_result["candidate_truncated"] <= task_result["baseline_truncated"]
            and task_result["candidate_empty"] <= task_result["baseline_empty"]
            and task_result["candidate_finite_logprobs"]
            and task_result["baseline_finite_logprobs"]
            and task_result["completed_format_invalid"]["candidate"]
            <= task_result["completed_format_invalid"]["baseline"]
        )
        output["tasks"][task] = task_result

    output["coverage"] = {
        "baseline_requests": sum(
            row["baseline_applied_calls"] == 0 for row in case_rows
        ),
        "candidate_requests": sum(
            row["candidate_applied_calls"] > 0 for row in case_rows
        ),
        "candidate_problem_overlap_requests": sum(
            row["candidate_applied_overlap_calls"] > 0 for row in case_rows
        ),
        "target_shape_pairs": sum(row["same_target_shapes"] for row in case_rows),
        "all_308_candidate_applied": all(
            row["candidate_applied_calls"] > 0 for row in case_rows
        ),
        "all_308_problem_overlap": all(
            row["candidate_applied_overlap_calls"] > 0 for row in case_rows
        ),
        "all_308_prompt_equal": all(row["prompt_equal"] for row in case_rows),
        "all_308_max_tokens_equal": all(row["max_tokens_equal"] for row in case_rows),
        "all_308_target_shapes_equal": all(
            row["same_target_shapes"] for row in case_rows
        ),
    }
    output["quality_gate_pass"] = all(
        task_result["score_gate_pass"] for task_result in output["tasks"].values()
    ) and all(output["coverage"].values())
    output["decision"] = (
        "additional quality gate passed; 32K retention may be evaluated"
        if output["quality_gate_pass"]
        else "additional quality gate not met; stop before 32K retention"
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["tasks"], indent=2))


if __name__ == "__main__":
    main()
