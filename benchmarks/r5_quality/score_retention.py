# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score the fixed 32K R5 content-retention diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def score_text(text: str, answer: str) -> dict[str, Any]:
    stripped = text.strip()
    return {
        "exact": stripped == answer,
        "contains_answer": answer in text,
        "empty": not stripped,
        "trimmed_text": stripped,
    }


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {row["case_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate case IDs in {path}")
    return result


def _coverage(row: dict[str, Any], mode: str) -> dict[str, Any]:
    hook = row.get("hook") or {}
    applied = int(hook.get("applied_calls", 0))
    overlap = int(hook.get("applied_overlap_calls", 0))
    return {
        "applied_calls": applied,
        "applied_overlap_calls": overlap,
        "layers": hook.get("layers", {}),
        "shapes": hook.get("shapes", {}),
        "expected": (applied == 0 if mode == "baseline" else applied > 0),
        "target_overlap": (overlap > 0 if mode == "candidate" else True),
    }


def score_records(
    suite_path: Path, manifest_path: Path, baseline_path: Path, candidate_path: Path
) -> dict[str, Any]:
    suite_raw = suite_path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    import hashlib

    suite_sha256 = hashlib.sha256(suite_raw).hexdigest()
    if suite_sha256 != manifest["suite_sha256"]:
        raise ValueError("suite hash does not match manifest")
    cases = {
        row["case_id"]: row
        for row in (json.loads(line) for line in suite_raw.splitlines())
    }
    baseline = _load_records(baseline_path)
    candidate = _load_records(candidate_path)
    if set(baseline) != set(cases) or set(candidate) != set(cases):
        raise ValueError("baseline/candidate case sets do not match suite")

    paired: list[dict[str, Any]] = []
    for case_id in sorted(cases):
        case = cases[case_id]
        left, right = baseline[case_id], candidate[case_id]
        if left["prompt_sha256"] != right["prompt_sha256"]:
            raise ValueError(f"prompt hash differs between arms: {case_id}")
        if left["prompt_sha256"] != case["prompt_sha256"]:
            raise ValueError(f"prompt hash differs from suite: {case_id}")
        base_score = score_text(left["text"], case["answer"])
        cand_score = score_text(right["text"], case["answer"])
        category = (
            "both_correct"
            if base_score["exact"] and cand_score["exact"]
            else "baseline_only"
            if base_score["exact"]
            else "candidate_only"
            if cand_score["exact"]
            else "both_incorrect"
        )
        paired.append(
            {
                "case_id": case_id,
                "content_id": case["content_id"],
                "position": case["position"],
                "identifier": case["identifier"],
                "answer": case["answer"],
                "category": category,
                "baseline": {
                    "exact": base_score["exact"],
                    "contains_answer": base_score["contains_answer"],
                    "empty": base_score["empty"],
                    "finish_reason": left.get("finish_reason"),
                    "truncated": left.get("finish_reason") == "length",
                    "output_tokens": len(left.get("output_token_ids", [])),
                    "elapsed_seconds": left.get("elapsed_seconds"),
                    "first_token_seconds": left.get("first_token_seconds"),
                    "coverage": _coverage(left, "baseline"),
                },
                "candidate": {
                    "exact": cand_score["exact"],
                    "contains_answer": cand_score["contains_answer"],
                    "empty": cand_score["empty"],
                    "finish_reason": right.get("finish_reason"),
                    "truncated": right.get("finish_reason") == "length",
                    "output_tokens": len(right.get("output_token_ids", [])),
                    "elapsed_seconds": right.get("elapsed_seconds"),
                    "first_token_seconds": right.get("first_token_seconds"),
                    "coverage": _coverage(right, "candidate"),
                },
            }
        )

    by_position: dict[str, dict[str, Any]] = {}
    for position in manifest["positions"]:
        rows = [row for row in paired if row["position"] == position]
        by_position[position] = {
            "cases": len(rows),
            "categories": dict(Counter(row["category"] for row in rows)),
            "baseline_correct": sum(row["baseline"]["exact"] for row in rows),
            "candidate_correct": sum(row["candidate"]["exact"] for row in rows),
            "baseline_truncated": sum(row["baseline"]["truncated"] for row in rows),
            "candidate_truncated": sum(row["candidate"]["truncated"] for row in rows),
        }

    coverage = {
        "baseline_all_unapplied": all(
            row["baseline"]["coverage"]["expected"] for row in paired
        ),
        "candidate_all_applied": all(
            row["candidate"]["coverage"]["expected"] for row in paired
        ),
        "candidate_all_target_overlap": all(
            row["candidate"]["coverage"]["target_overlap"] for row in paired
        ),
    }
    resource_anomalies = []
    for mode, rows in (("baseline", baseline), ("candidate", candidate)):
        for case_id, row in rows.items():
            if row.get("stream_errors"):
                resource_anomalies.append(f"{mode}/{case_id}:stream_error")
            if row.get("empty"):
                resource_anomalies.append(f"{mode}/{case_id}:empty")
            before = row.get("resources_before", {}).get("memory_events", {})
            after = row.get("resources_after", {}).get("memory_events", {})
            for key in ("oom", "oom_kill", "oom_group_kill"):
                if after.get(key, 0) > before.get(key, 0):
                    resource_anomalies.append(f"{mode}/{case_id}:new_{key}")
    summary = {
        "cases": len(paired),
        "categories": dict(Counter(row["category"] for row in paired)),
        "baseline_correct": sum(row["baseline"]["exact"] for row in paired),
        "candidate_correct": sum(row["candidate"]["exact"] for row in paired),
        "baseline_truncated": sum(row["baseline"]["truncated"] for row in paired),
        "candidate_truncated": sum(row["candidate"]["truncated"] for row in paired),
        "coverage": coverage,
        "resource_anomalies": resource_anomalies,
        "main_exact_match_only": True,
        "position_results": by_position,
        "diagnostic_only": True,
        "historical_quality_gate_unchanged": True,
        "production_adoption": "not authorized",
    }
    return {
        "suite_id": manifest["suite_id"],
        "suite_sha256": suite_sha256,
        "summary": summary,
        "cases": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_records(args.suite, args.manifest, args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
