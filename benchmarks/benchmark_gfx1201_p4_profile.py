# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze the P4 first-chunk/continuation split of a gfx1201 trace.

This helper is deliberately separate from the P1 analyzer.  P1 predates the
``execute_context_1(<chunk>)`` annotations and classifies Math SDPA Cijk
kernels as dense GEMM.  P4 needs the first context (raw K/V) separated from
later continuation contexts, and it uses runtime-launch correlations to
attribute Math SDPA kernels without relying on GPU timestamp overlap.

The analyzer never launches a model or changes dispatch.  It streams a gzip
trace with ``ijson`` and writes a machine-readable report suitable for the
P4.0 design record.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ijson
import regex as re

P4_FAMILIES = (
    "first_chunk_raw_attention",
    "continuation_attention",
    "mxfp4_dequant_dense_gemm",
    "turboquant_store",
    "gdn_fla_prefill",
    "norm_activation_indexing",
    "copies_conversions",
    "cached_prefix_dequant",
    "other",
)

_EXECUTE_CONTEXT = re.compile(r"^execute_context_1\((\d+)\)")


def _trace_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield trace events without materializing the JSON document."""

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as trace_file:
        yield from ijson.items(trace_file, "traceEvents.item")


def _kernel_family(name: str) -> str:
    """Classify a non-attention kernel into one P4 cost center."""

    lowered = name.lower()
    if any(
        token in lowered for token in ("mxfp4", "cijk_", "wvsplitk", "gemm", "gemv")
    ):
        return "mxfp4_dequant_dense_gemm"
    if "_tq_fused_store" in lowered or (
        "tq" in lowered and "store" in lowered and "cache" in lowered
    ):
        return "turboquant_store"
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
    if any(
        token in lowered
        for token in (
            "dequant",
            "k8v4",
            "turboquant",
            "bfloat16tofloat32",
            "float32tobfloat16",
        )
    ):
        return "cached_prefix_dequant"
    if any(
        token in lowered
        for token in (
            "copy",
            "cat",
            "convert",
            "contig",
            "transpose",
            "copybuffer",
        )
    ):
        return "copies_conversions"
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


def _is_attention_kernel(name: str) -> bool:
    """Return whether a kernel name is an attention implementation."""

    lowered = name.lower()
    return any(
        token in lowered
        for token in (
            "tq_unified_attention",
            "paged_attention",
            "flash_attn",
            "flashattention",
            "scaled_dot_product",
            "attention",
        )
    )


def _scope_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Collect first-chunk/continuation scopes and count trace events."""

    rows: list[dict[str, Any]] = []
    event_count = 0
    for event in _trace_events(path):
        event_count += 1
        if event.get("cat") != "user_annotation":
            continue
        name = str(event.get("name", ""))
        match = _EXECUTE_CONTEXT.match(name)
        if match is None:
            continue
        start = float(event["ts"])
        rows.append(
            {
                "name": name,
                "ts": start,
                "end": start + float(event.get("dur", 0.0)),
                "q_len": int(match.group(1)),
            }
        )
    rows.sort(key=lambda row: (row["ts"], row["end"]))
    for ordinal, row in enumerate(rows):
        row["ordinal"] = ordinal
        row["phase"] = "first_chunk" if ordinal == 0 else "continuation"
    return rows, event_count


def _cpu_attention_ops(
    path: Path, scopes: list[dict[str, Any]]
) -> tuple[
    dict[int, list[tuple[float, float]]], dict[int, dict[str, int]], dict[int, float]
]:
    """Collect outer SDPA intervals and operation counts for every scope."""

    intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    inclusive_us: dict[int, float] = defaultdict(float)
    starts = [float(row["ts"]) for row in scopes]
    ends = [float(row["end"]) for row in scopes]
    for event in _trace_events(path):
        if event.get("cat") != "cpu_op":
            continue
        timestamp = float(event.get("ts", 0.0))
        duration = float(event.get("dur", 0.0))
        event_end = timestamp + duration
        scope_index = bisect.bisect_right(starts, timestamp) - 1
        if scope_index < 0 or event_end > ends[scope_index]:
            continue
        name = str(event.get("name", ""))
        if name not in (
            "aten::scaled_dot_product_attention",
            "aten::_scaled_dot_product_attention_math",
        ):
            continue
        counts[scope_index][name] += 1
        if name == "aten::scaled_dot_product_attention":
            intervals[scope_index].append((timestamp, event_end))
            inclusive_us[scope_index] += duration
    for scope_intervals in intervals.values():
        scope_intervals.sort()
    return (
        dict(intervals),
        {index: dict(scope_counts) for index, scope_counts in counts.items()},
        dict(inclusive_us),
    )


def _runtime_correlations(
    path: Path,
    scopes: list[dict[str, Any]],
    attention_intervals: dict[int, list[tuple[float, float]]],
) -> tuple[dict[int, int], set[tuple[int, int]], int]:
    """Map launches to execute scopes and all outer SDPA operations."""

    starts = [float(row["ts"]) for row in scopes]
    scope_by_correlation: dict[int, int] = {}
    attention_correlations: set[tuple[int, int]] = set()
    runtime_count = 0
    interval_starts = {
        scope_index: [interval[0] for interval in intervals]
        for scope_index, intervals in attention_intervals.items()
    }
    for event in _trace_events(path):
        if event.get("cat") != "cuda_runtime":
            continue
        args = event.get("args", {})
        correlation = args.get("correlation")
        if correlation is None:
            continue
        runtime_count += 1
        timestamp = float(event.get("ts", 0.0))
        scope_index = bisect.bisect_right(starts, timestamp) - 1
        if scope_index < 0 or timestamp > float(scopes[scope_index]["end"]):
            continue
        correlation = int(correlation)
        scope_by_correlation[correlation] = scope_index
        starts_for_scope = interval_starts.get(scope_index, [])
        if not starts_for_scope:
            continue
        intervals = attention_intervals[scope_index]
        interval_index = bisect.bisect_right(starts_for_scope, timestamp) - 1
        if interval_index >= 0 and timestamp <= intervals[interval_index][1]:
            attention_correlations.add((scope_index, correlation))
    return scope_by_correlation, attention_correlations, runtime_count


def analyze_trace(path: Path) -> dict[str, Any]:
    """Return the P4 first/continuation cost-center report."""

    scopes, event_count = _scope_rows(path)
    if not scopes:
        raise ValueError("trace has no execute_context_1 prefill scopes")
    attention_intervals, cpu_ops_by_scope, sdpa_cpu_us_by_scope = _cpu_attention_ops(
        path, scopes
    )
    correlations, attention_correlations, runtime_count = _runtime_correlations(
        path, scopes, attention_intervals
    )

    totals: dict[str, list[float | int]] = {family: [0.0, 0] for family in P4_FAMILIES}
    per_scope: dict[int, list[float | int]] = defaultdict(lambda: [0.0, 0])
    kernel_count = 0
    out_of_scope = 0
    raw_kernel_names: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    continuation_attention_names: dict[str, list[float | int]] = defaultdict(
        lambda: [0.0, 0]
    )

    for event in _trace_events(path):
        if event.get("cat") != "kernel":
            continue
        kernel_count += 1
        correlation = event.get("args", {}).get("correlation")
        if correlation is None or int(correlation) not in correlations:
            out_of_scope += 1
            continue
        correlation = int(correlation)
        scope_index = correlations[correlation]
        name = str(event.get("name", ""))
        duration = float(event.get("dur", 0.0))
        attention_key = (scope_index, correlation)
        if attention_key in attention_correlations and scope_index == 0:
            family = "first_chunk_raw_attention"
            row = raw_kernel_names[name]
        elif attention_key in attention_correlations or (
            scope_index > 0 and _is_attention_kernel(name)
        ):
            family = "continuation_attention"
            row = continuation_attention_names[name]
        else:
            family = _kernel_family(name)
            row = None
        totals[family][0] += duration
        totals[family][1] += 1
        per_scope[scope_index][0] += duration
        per_scope[scope_index][1] += 1
        if row is not None:
            row[0] += duration
            row[1] += 1

    total_us = sum(float(row[0]) for row in totals.values())
    family_rows = []
    for family in P4_FAMILIES:
        duration_us, calls = totals[family]
        family_rows.append(
            {
                "family": family,
                "gpu_ms": float(duration_us) / 1000.0,
                "share_percent": (
                    100.0 * float(duration_us) / total_us if total_us else 0.0
                ),
                "calls": int(calls),
            }
        )
    family_rows.sort(key=lambda row: row["gpu_ms"], reverse=True)
    first_raw_ms = float(totals["first_chunk_raw_attention"][0]) / 1000.0
    first_cpu_ops = cpu_ops_by_scope.get(0, {})
    first_sdpa_cpu_us = sdpa_cpu_us_by_scope.get(0, 0.0)
    continuation_cpu_ops: dict[str, int] = defaultdict(int)
    continuation_sdpa_cpu_us = 0.0
    for scope_index in range(1, len(scopes)):
        for name, count in cpu_ops_by_scope.get(scope_index, {}).items():
            continuation_cpu_ops[name] += count
        continuation_sdpa_cpu_us += sdpa_cpu_us_by_scope.get(scope_index, 0.0)
    return {
        "schema_version": 2,
        "trace": str(path),
        "trace_event_count": event_count,
        "runtime_event_count": runtime_count,
        "kernel_count": kernel_count,
        "out_of_scope_kernel_count": out_of_scope,
        "scoped_kernel_count": int(sum(int(row[1]) for row in totals.values())),
        "scope_count": len(scopes),
        "scopes": [
            {
                "ordinal": row["ordinal"],
                "name": row["name"],
                "phase": row["phase"],
                "q_len": row["q_len"],
            }
            for row in scopes
        ],
        "prefill_q_tokens": sum(int(row["q_len"]) for row in scopes),
        "first_chunk": {
            "q_len": int(scopes[0]["q_len"]),
            "outer_sdpa_calls": len(attention_intervals.get(0, [])),
            "cpu_ops": first_cpu_ops,
            "sdpa_cpu_inclusive_ms": first_sdpa_cpu_us / 1000.0,
            "raw_attention_launch_correlations": sum(
                1 for scope_index, _ in attention_correlations if scope_index == 0
            ),
            "raw_attention_kernel_ms": first_raw_ms,
            "raw_attention_kernel_calls": int(totals["first_chunk_raw_attention"][1]),
            "raw_attention_kernel_names": {
                name: {"gpu_ms": values[0] / 1000.0, "calls": int(values[1])}
                for name, values in sorted(
                    raw_kernel_names.items(), key=lambda item: -item[1][0]
                )
            },
        },
        "continuation": {
            "scope_count": max(0, len(scopes) - 1),
            "q_tokens": sum(int(row["q_len"]) for row in scopes[1:]),
            "outer_sdpa_calls": sum(
                len(attention_intervals.get(scope_index, []))
                for scope_index in range(1, len(scopes))
            ),
            "cpu_ops": dict(continuation_cpu_ops),
            "sdpa_cpu_inclusive_ms": continuation_sdpa_cpu_us / 1000.0,
            "math_sdpa_ops": continuation_cpu_ops.get(
                "aten::_scaled_dot_product_attention_math", 0
            ),
            "attention_launch_correlations": sum(
                1 for scope_index, _ in attention_correlations if scope_index > 0
            ),
            "attention_kernel_ms": float(totals["continuation_attention"][0]) / 1000.0,
            "attention_kernel_calls": int(totals["continuation_attention"][1]),
            "attention_kernel_names": {
                name: {"gpu_ms": values[0] / 1000.0, "calls": int(values[1])}
                for name, values in sorted(
                    continuation_attention_names.items(),
                    key=lambda item: -item[1][0],
                )
            },
        },
        "kernel_sum_ms": total_us / 1000.0,
        "family_totals": family_rows,
        "scope_kernel_totals": [
            {
                "ordinal": ordinal,
                "phase": scopes[ordinal]["phase"],
                "q_len": scopes[ordinal]["q_len"],
                "gpu_ms": values[0] / 1000.0,
                "calls": int(values[1]),
            }
            for ordinal, values in sorted(per_scope.items())
        ],
        "gate": {
            "raw_attention_share_percent": (
                100.0 * first_raw_ms / (total_us / 1000.0) if total_us else 0.0
            ),
            "first_raw_attention_ge_10_percent": (
                100.0 * first_raw_ms / (total_us / 1000.0) >= 10.0
                if total_us
                else False
            ),
            "first_chunk_math_sdpa_fallback": first_cpu_ops.get(
                "aten::_scaled_dot_product_attention_math", 0
            )
            > 0,
            "status": (
                "pass"
                if first_cpu_ops.get("aten::_scaled_dot_product_attention_math", 0) > 0
                or (total_us > 0 and 100.0 * first_raw_ms / (total_us / 1000.0) >= 10.0)
                else "stop"
            ),
            "reason": (
                "first chunk uses explicit Math SDPA fallback"
                if first_cpu_ops.get("aten::_scaled_dot_product_attention_math", 0) > 0
                else "first-chunk raw attention is below the 10% residual-time gate"
            ),
        },
    }


def _load_requests(path: Path) -> list[dict[str, Any]]:
    with path.open() as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = analyze_trace(args.trace)
    if args.requests:
        result["requests"] = _load_requests(args.requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(result["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
