# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize an R5 quality-stop run without executing more model requests."""

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.quality_report.read_text())
    if report["pass"]:
        raise ValueError("This finalizer only records a quality stop")
    folder = args.directory / "quality-validated-salt"
    arms = {
        mode: read_jsonl(folder / f"{mode}.jsonl") for mode in ("baseline", "candidate")
    }
    resources = read_jsonl(args.directory / "resources-verified.jsonl")
    output = {
        "base": "d5fec6675f832ba6bf753247acd092ef86e5cd56",
        "decision": "quality gate not met; stop without production adoption",
        "status": {
            "cold32k_ttft": "historical pass preserved: 2.1733x",
            "strict_math_diagnostic": "historical failure preserved",
            "quality_308": "fail: new syntax-invalid output; task scores noninferior",
            "retention_32k": "not evaluated: stopped at quality gate",
            "natural_decode_mtp": "not evaluated: stopped at quality gate",
            "fixed_state_decode": "not evaluated: no suitable full-state replay found",
            "prefix_reuse": "not evaluated: stopped at quality gate",
            "sequential_soak": "not evaluated: stopped at quality gate",
        },
        "quality": report,
        "run": {},
        "resource_summary": {
            "samples": len(resources),
            "sampled_ram_peak_bytes": max(r["memory_current"] for r in resources),
            "sampled_shm_peak_bytes": max(r["shm_used_bytes"] for r in resources),
            "oom_event_deltas": {
                name: resources[-1]["memory_events"][name]
                - resources[0]["memory_events"][name]
                for name in ("oom", "oom_kill", "oom_group_kill")
            },
            "sampled_ram_final_bytes": resources[-1]["memory_current"],
            "note": "sampled values, not exact peaks; startup and shutdown included",
        },
        "production_adopted": False,
        "source_changes": "benchmarks/tests/docs only; attention math unchanged",
        "judge": "stdin-open isolated execution; original no-stdin results invalid",
    }
    for mode, rows in arms.items():
        assert len(rows) == 308
        assert len({r["run_id"] for r in rows}) == 308
        for row in rows:
            usage = row["response"]["usage"]
            assert len(row["prompt_token_ids"]) == usage["prompt_tokens"]
            assert len(row["output_token_ids"]) == usage["completion_tokens"]
            hook = row["hook"]
            assert hook["run_id"] == row["run_id"]
            assert hook["calls"] == sum(hook["layers"].values())
            assert hook["calls"] == sum(hook["shapes"].values())
            for shape in hook["shapes"]:
                cached, query = shape.removeprefix("cached").split("_q")
                assert int(cached) > 0 and int(query) > 128
            assert all(
                name.startswith("language_model.model.layers.")
                for name in hook["layers"]
            )
        output["run"][mode] = {
            "requests": len(rows),
            "request_and_scope_invariants": "passed",
            "output_tokens": sum(len(r["output_token_ids"]) for r in rows),
            "applied_requests": sum(r["hook"]["applied_calls"] > 0 for r in rows),
            "applied_calls": sum(r["hook"]["applied_calls"] for r in rows),
            "finite_logprobs": all(r["finite_logprobs"] for r in rows),
            "completion_reasons": sorted({r["finish_reason"] for r in rows}),
            "response_sample_ram_min_bytes": min(
                r["resources"]["memory_current"] for r in rows
            ),
            "response_sample_ram_max_bytes": max(
                r["resources"]["memory_current"] for r in rows
            ),
            "gpu_allocator_lifetime_peak_allocated": max(
                r["hook"]["peak_allocated_bytes"] for r in rows
            ),
            "gpu_allocator_lifetime_peak_reserved": max(
                r["hook"]["peak_reserved_bytes"] for r in rows
            ),
        }
    with args.output.open("x") as target:
        json.dump(output, target, indent=2)
        target.write("\n")


if __name__ == "__main__":
    main()
