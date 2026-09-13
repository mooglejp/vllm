# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locate the first 4K attention difference in diagnostic JSONL traces.

This CPU-only tool aligns separately served runs by continuation call order.
It never runs a kernel and does not infer a model-wide conclusion from one
layer; the report is explicitly scoped to the captured 4K request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FULL_PREFIX = "language_model.model.layers."
INPUT_FIELDS = ("query_digest", "key_chunk_digest", "value_chunk_digest")


def load(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not records:
        raise ValueError(f"{path} is empty")
    return records


def full_attention(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["layer"].startswith(FULL_PREFIX)]


def align(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], label: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(baseline) != len(candidate):
        raise ValueError(
            f"{label}: record count differs ({len(baseline)} vs {len(candidate)})"
        )
    pairs = []
    for expected, actual in zip(baseline, candidate, strict=True):
        metadata = (
            "trace_index",
            "layer",
            "layer_index",
            "cached_len",
            "q_len",
            "seq_len",
        )
        mismatches = {
            field: (expected[field], actual[field])
            for field in metadata
            if expected[field] != actual[field]
        }
        if mismatches:
            raise ValueError(
                f"{label}: trace order differs at index "
                f"{expected['trace_index']}: {mismatches}"
            )
        pairs.append((expected, actual))
    return pairs


def first_mismatch(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any] | None:
    for baseline, candidate in pairs:
        if baseline["output_digest"] == candidate["output_digest"]:
            continue
        inputs_equal = all(
            baseline[field] == candidate[field] for field in INPUT_FIELDS
        )
        return {
            "trace_index": baseline["trace_index"],
            "chunk": {
                "cached_len": baseline["cached_len"],
                "q_len": baseline["q_len"],
                "seq_len": baseline["seq_len"],
            },
            "layer": baseline["layer"],
            "layer_index": baseline["layer_index"],
            "operation": {
                "baseline": baseline["operation"],
                "candidate": candidate["operation"],
            },
            "input_digests_equal": inputs_equal,
            "input_digest_equal_fields": [
                field for field in INPUT_FIELDS if baseline[field] == candidate[field]
            ],
            "output_digest": {
                "baseline": baseline["output_digest"],
                "candidate": candidate["output_digest"],
            },
            "output_stats": {
                "baseline": baseline["output_stats"],
                "candidate": candidate["output_stats"],
            },
            "cause_class": (
                "attention_operation"
                if inputs_equal
                else "upstream_input_or_attention_operation"
            ),
        }
    return None


def first_input_mismatch(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any] | None:
    for baseline, candidate in pairs:
        differing = [
            field for field in INPUT_FIELDS if baseline[field] != candidate[field]
        ]
        if differing:
            return {
                "trace_index": baseline["trace_index"],
                "chunk": {
                    "cached_len": baseline["cached_len"],
                    "q_len": baseline["q_len"],
                    "seq_len": baseline["seq_len"],
                },
                "layer": baseline["layer"],
                "layer_index": baseline["layer_index"],
                "differing_input_digests": differing,
            }
    return None


def compare(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    pairs = align(
        full_attention(baseline_records), full_attention(candidate_records), label
    )
    return {
        "scope": "language_model_full_attention_only",
        "record_count": len(pairs),
        "first_output_difference": first_mismatch(pairs),
        "first_input_difference": first_input_mismatch(pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--pvfp32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--revision", default="8fb487f41266db2e9ba634632dc3cf99e26d8704"
    )
    args = parser.parse_args()

    baseline = load(args.baseline)
    bf16 = load(args.bf16)
    pvfp32 = load(args.pvfp32)
    result = {
        "revision": args.revision,
        "scope": {
            "request": "one independently served 4K request per variant",
            "full_attention_layer_prefix": FULL_PREFIX,
            "candidate_output_chained_into_baseline": False,
        },
        "trace_counts": {
            "baseline": len(baseline),
            "bf16": len(bf16),
            "pvfp32": len(pvfp32),
        },
        "operations": {
            "baseline": sorted({record["operation"] for record in baseline}),
            "bf16": sorted({record["operation"] for record in bf16}),
            "pvfp32": sorted({record["operation"] for record in pvfp32}),
        },
        "baseline_vs_bf16": compare(baseline, bf16, "baseline_vs_bf16"),
        "baseline_vs_pvfp32": compare(baseline, pvfp32, "baseline_vs_pvfp32"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
