# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze a gfx1201 long-prefill baseline trace.

The P1 profile can produce multi-gigabyte Chrome traces.  Loading one with
``json.load`` makes the trace itself the memory bottleneck, so this helper
uses ``ijson`` and keeps only CPU scopes, runtime correlations, and aggregate
kernel totals.  It does not launch a server or change runtime dispatch.

Example::

    .venv/bin/python benchmarks/benchmark_gfx1201_long_prefill.py \
        --trace /tmp/p1/rank0.pt.trace.json.gz \
        --requests /tmp/p1/chunk128.jsonl \
        --output /tmp/p1/summary.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ijson

PREFILL_FAMILIES = (
    "mxfp4_weight_dequant_dense_gemm",
    "turboquant_store",
    "cached_prefix_dequant",
    "full_kv_copy_conversion",
    "attention_compute",
    "gdn_fla_prefill",
    "norm_activation_indexing",
    "other",
)


def _trace_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield trace events without materializing the trace JSON."""

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as trace_file:
        yield from ijson.items(trace_file, "traceEvents.item")


def _kernel_family(name: str) -> str:
    """Assign a kernel name to one mutually exclusive P1 cost center."""

    lowered = name.lower()
    if any(
        token in lowered for token in ("mxfp4", "cijk_", "wvsplitk", "gemm", "gemv")
    ):
        return "mxfp4_weight_dequant_dense_gemm"
    if "_tq_fused_store" in lowered or (
        "tq" in lowered and "store" in lowered and "cache" in lowered
    ):
        return "turboquant_store"
    if any(
        token in lowered
        for token in (
            "tq_unified_attention",
            "paged_attention",
            "flash_attn",
            "flashattention",
            "scaled_dot_product",
            "attention",
        )
    ):
        return "attention_compute"
    if any(
        token in lowered
        for token in (
            "chunk_",
            "gated_delta",
            "gdn",
            "conv1d",
            "post_conv",
            "recompute_w_u",
            "merge_16x16",
            "solve_tril",
        )
    ):
        return "gdn_fla_prefill"
    if any(token in lowered for token in ("dequant", "k8v4", "turboquant")):
        return "cached_prefix_dequant"
    if any(
        token in lowered
        for token in (
            "copy",
            "cat",
            "convert",
            "contig",
            "transpose",
            "bfloat16tofloat32",
            "float32tobfloat16",
            "copybuffer",
        )
    ):
        return "full_kv_copy_conversion"
    if any(
        token in lowered
        for token in (
            "norm",
            "rms",
            "silu",
            "gelu",
            "sigmoid",
            "rsqrt",
            "pow_",
            "index",
            "scatter",
            "gather",
            "arange",
            "masked",
            "fill",
            "clamp",
            "where",
            "add<",
            "mul<",
            "div<",
        )
    ):
        return "norm_activation_indexing"
    return "other"


def _phase_owner(scopes: list[dict[str, Any]]) -> tuple[str, str]:
    """Resolve the phase and owner from a runtime launch's scope stack."""

    names = [scope["name"] for scope in scopes]
    phase = "outside"
    owner = "outside"
    for name in names:
        if name.endswith(".target.forward"):
            parts = name.split(".")
            phase = parts[1] if len(parts) > 1 else "unknown"
    if any(name.endswith(".drafter") for name in names):
        owner = "drafter"
    elif any(name.endswith(".target.forward") for name in names):
        owner = "target_forward"
    elif "tq.target.sample" in names:
        owner = "target_sample"
    elif names:
        owner = "runner"
    return phase, owner


def _collect_scopes(
    path: Path,
) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], int]:
    scopes: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    event_count = 0
    for event in _trace_events(path):
        event_count += 1
        if event.get("cat") != "user_annotation":
            continue
        name = str(event.get("name", ""))
        if not name.startswith("tq."):
            continue
        scope = {
            "name": name,
            "ts": float(event["ts"]),
            "end": float(event["ts"]) + float(event.get("dur", 0.0)),
        }
        scopes[(int(event["pid"]), int(event["tid"]))].append(scope)
    for rows in scopes.values():
        rows.sort(key=lambda row: (row["ts"], -row["end"]))
    return scopes, event_count


def _collect_correlations(
    path: Path,
    scopes: dict[tuple[int, int], list[dict[str, Any]]],
) -> tuple[dict[int, tuple[str, str]], int]:
    correlations: dict[int, tuple[str, str]] = {}
    state: dict[tuple[int, int], tuple[int, list[dict[str, Any]]]] = {}
    runtime_count = 0
    for event in _trace_events(path):
        if event.get("cat") != "cuda_runtime":
            continue
        args = event.get("args", {})
        if "correlation" not in args:
            continue
        runtime_count += 1
        key = (int(event["pid"]), int(event["tid"]))
        rows = scopes.get(key, [])
        index, stack = state.get(key, (0, []))
        timestamp = float(event["ts"])
        while index < len(rows) and rows[index]["ts"] <= timestamp:
            row = rows[index]
            while stack and stack[-1]["end"] <= row["ts"]:
                stack.pop()
            stack.append(row)
            index += 1
        while stack and stack[-1]["end"] <= timestamp:
            stack.pop()
        state[key] = (index, stack.copy())
        correlations[int(args["correlation"])] = _phase_owner(stack)
    return correlations, runtime_count


def analyze_trace(path: Path) -> dict[str, Any]:
    """Return aggregate phase/cost-center data from a Chrome trace."""

    scopes, event_count = _collect_scopes(path)
    correlations, runtime_count = _collect_correlations(path, scopes)
    cells: dict[tuple[str, str, str], list[float | int]] = defaultdict(lambda: [0.0, 0])
    phase_totals: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    kernel_count = 0
    uncorrelated = 0
    for event in _trace_events(path):
        if event.get("cat") != "kernel":
            continue
        kernel_count += 1
        correlation = event.get("args", {}).get("correlation")
        phase, owner = correlations.get(correlation, ("uncorrelated", "unknown"))
        family = _kernel_family(str(event.get("name", "")))
        duration = float(event.get("dur", 0.0))
        cells[phase, owner, family][0] += duration
        cells[phase, owner, family][1] += 1
        phase_totals[phase][0] += duration
        phase_totals[phase][1] += 1
        if phase == "uncorrelated":
            uncorrelated += 1

    phases: dict[str, dict[str, Any]] = {}
    for phase, (duration, count) in sorted(phase_totals.items()):
        rows = [
            {
                "owner": owner,
                "family": family,
                "gpu_ms": cells[phase, owner, family][0] / 1000,
                "calls": int(cells[phase, owner, family][1]),
            }
            for (cell_phase, owner, family) in cells
            if cell_phase == phase
        ]
        family_totals: dict[str, dict[str, float | int]] = {
            family: {"family": family, "gpu_ms": 0.0, "calls": 0}
            for family in PREFILL_FAMILIES
        }
        for row in rows:
            family_total = family_totals[row["family"]]
            family_total["gpu_ms"] = float(family_total["gpu_ms"]) + float(
                row["gpu_ms"]
            )
            family_total["calls"] = int(family_total["calls"]) + int(row["calls"])
        family_rows = sorted(
            family_totals.values(), key=lambda row: row["gpu_ms"], reverse=True
        )
        rows.sort(key=lambda row: row["gpu_ms"], reverse=True)
        phases[phase] = {
            "kernel_sum_ms": duration / 1000,
            "kernel_count": int(count),
            "cells": rows,
            "family_totals": family_rows,
            "largest_cost_centers": rows[:2],
        }
    return {
        "trace": str(path),
        "trace_event_count": event_count,
        "runtime_correlation_count": runtime_count,
        "kernel_count": kernel_count,
        "uncorrelated_kernel_count": uncorrelated,
        "all_kernels_correlated": uncorrelated == 0,
        "phases": phases,
        "scope_counts": {
            name: sum(
                1 for rows in scopes.values() for row in rows if row["name"] == name
            )
            for name in sorted(
                {row["name"] for rows in scopes.values() for row in rows}
            )
        },
    }


def _load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as input_file:
            rows.extend(json.loads(line) for line in input_file if line.strip())
    return rows


def _request_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row.get("phase") == "measured"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measured:
        grouped[str(row["prompt_tokens"])].append(row)
    contexts = {}
    for context, context_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        contexts[context] = {
            "samples": len(context_rows),
            "labels": sorted({str(row.get("label", "")) for row in context_rows}),
            "prompt_sha256": sorted({row.get("prompt_sha256") for row in context_rows}),
            "ttft_s": [row.get("ttft_s") for row in context_rows],
            "prefill_tokens_s": [
                row["prompt_tokens"] / row["ttft_s"] for row in context_rows
            ],
            "e2e_tokens_s": [row.get("e2e_tokens_s") for row in context_rows],
            "decode_tokens_s": [row.get("decode_tokens_s") for row in context_rows],
            "completion_tokens": [row.get("completion_tokens") for row in context_rows],
        }
    return {"measured_rows": len(measured), "contexts": contexts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--requests", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "schema_version": 1,
        "trace_profile": analyze_trace(args.trace),
    }
    if args.requests:
        result["request_profile"] = _request_summary(_load_jsonl(args.requests))
    prefill = result["trace_profile"]["phases"].get("prefill", {})
    result["p1_gate"] = {
        "status": (
            "pass"
            if prefill
            and prefill.get("kernel_count", 0) > 0
            and result["trace_profile"]["all_kernels_correlated"]
            and len(prefill.get("largest_cost_centers", [])) >= 2
            else "incomplete"
        ),
        "largest_prefill_cost_centers": prefill.get("largest_cost_centers", []),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result["p1_gate"], indent=2))


if __name__ == "__main__":
    main()
