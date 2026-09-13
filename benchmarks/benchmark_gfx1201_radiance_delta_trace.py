# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize a Radiance torch profiler trace by major execution phase.

The helper is intentionally an offline analyzer.  It does not launch a model
or change any dispatch setting.  Radiance emits ``execute_context_*`` GPU
annotations, so kernels are assigned to the smallest enclosing annotation and
then grouped by a name-based cost-center classifier.  Kernel durations are
summed; they are not a wall-clock critical-path measurement.

This is a control-comparison aid rather than a replacement for the fork's
launch-correlation analyzer.  The classifier is kept explicit in the output so
that a category change cannot be mistaken for a new measurement.
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

CATEGORIES = (
    "attention",
    "mxfp4_dequant_dense_gemm",
    "gdn_fla_prefill",
    "turboquant_store",
    "norm_activation_indexing",
    "copies_conversions",
    "other",
)

_CONTEXT = re.compile(r"^execute_context_([01])\((\d+)\)")


def _trace_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield trace events without materializing the JSON document."""

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as trace_file:
        yield from ijson.items(trace_file, "traceEvents.item")


def _kernel_category(name: str) -> str:
    """Map a Radiance kernel name to one diagnostic cost center."""

    lowered = name.lower()
    if any(
        token in lowered
        for token in (
            "unified_attention",
            "paged_attention",
            "flash_attn",
            "flashattention",
            "scaled_dot_product",
            "wvsplitk",
            "reduce_segments",
            "attention",
        )
    ):
        return "attention"
    if any(
        token in lowered
        for token in (
            "mxfp4",
            "cijk_",
            "fp8_gemm",
            "gemm",
            "gemv",
            "dynamic_per_token_scaled_fp8_quant",
            "fp8_quant",
        )
    ):
        return "mxfp4_dequant_dense_gemm"
    if any(
        token in lowered
        for token in (
            "gdn",
            "gated_delta",
            "fused_recurrent",
            "causal_conv1d",
            "conv_prep",
            "chunk_scan",
            "kkt_solve",
            "solve_tril",
            "recompute_w_u",
        )
    ):
        return "gdn_fla_prefill"
    if any(
        token in lowered
        for token in (
            "reshape_and_cache",
            "zero_kv_blocks",
            "turboquant_store",
            "tq_fused_store",
        )
    ):
        return "turboquant_store"
    if any(
        token in lowered
        for token in (
            "rms_norm",
            "layer_norm",
            "norm",
            "silu",
            "gelu",
            "sigmoid",
            "rsqrt",
            "masked_fill",
            "index",
            "scatter",
            "gather",
            "arange",
            "clamp",
            "where",
        )
    ):
        return "norm_activation_indexing"
    if any(
        token in lowered
        for token in (
            "copy",
            "memcpy",
            "contig",
            "transpose",
            "convert",
            "cat<",
            "fillbuffer",
            "batch_memcpy",
        )
    ):
        return "copies_conversions"
    return "other"


def _contexts(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Collect the GPU context intervals used for phase attribution."""

    contexts: list[dict[str, Any]] = []
    event_count = 0
    for event in _trace_events(path):
        event_count += 1
        if event.get("cat") != "gpu_user_annotation" or event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        match = _CONTEXT.match(name)
        if match is None:
            continue
        start = float(event.get("ts", 0.0))
        duration = float(event.get("dur", 0.0))
        mode = match.group(1)
        contexts.append(
            {
                "name": name,
                "ts": start,
                "end": start + duration,
                "duration_us": duration,
                "mode": mode,
                "phase": "prefill" if mode == "1" else "decode",
                "q_len": int(match.group(2)),
            }
        )
    contexts.sort(key=lambda row: (row["ts"], row["end"]))
    prefill_ordinal = 0
    for ordinal, row in enumerate(contexts):
        row["ordinal"] = ordinal
        if row["phase"] == "prefill":
            row["prefill_ordinal"] = prefill_ordinal
            row["subphase"] = "first_chunk" if prefill_ordinal == 0 else "continuation"
            prefill_ordinal += 1
        else:
            row["subphase"] = "decode"
    return contexts, event_count


def _containing_context(
    contexts: list[dict[str, Any]], starts: list[float], timestamp: float
) -> int | None:
    """Return the smallest context containing a kernel midpoint."""

    index = bisect.bisect_right(starts, timestamp) - 1
    candidates = range(max(0, index - 1), min(len(contexts), index + 3))
    containing = [
        candidate
        for candidate in candidates
        if contexts[candidate]["ts"] <= timestamp <= contexts[candidate]["end"]
    ]
    if not containing:
        return None
    return min(containing, key=lambda candidate: contexts[candidate]["end"])


def _empty_totals() -> dict[str, list[float | int]]:
    return {category: [0.0, 0] for category in CATEGORIES}


def _rows(totals: dict[str, list[float | int]]) -> list[dict[str, Any]]:
    total_us = sum(float(values[0]) for values in totals.values())
    rows = []
    for category in CATEGORIES:
        duration_us, calls = totals[category]
        rows.append(
            {
                "category": category,
                "gpu_ms_sum": float(duration_us) / 1000.0,
                "share_percent": (
                    100.0 * float(duration_us) / total_us if total_us else 0.0
                ),
                "calls": int(calls),
            }
        )
    return sorted(rows, key=lambda row: row["gpu_ms_sum"], reverse=True)


def analyze_trace(path: Path) -> dict[str, Any]:
    """Return a phase and cost-center summary for ``path``."""

    contexts, trace_event_count = _contexts(path)
    if not contexts:
        raise ValueError("trace has no execute_context GPU annotations")
    starts = [float(row["ts"]) for row in contexts]
    totals = _empty_totals()
    phase_totals: dict[str, dict[str, list[float | int]]] = defaultdict(_empty_totals)
    context_totals: dict[int, list[float | int]] = defaultdict(lambda: [0.0, 0])
    top_kernels: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    scoped_kernel_count = 0
    unscoped_kernel_count = 0
    unscoped_kernel_us = 0.0
    kernel_event_count = 0

    for event in _trace_events(path):
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        kernel_event_count += 1
        start = float(event.get("ts", 0.0))
        duration = float(event.get("dur", 0.0))
        context_index = _containing_context(contexts, starts, start + 0.5 * duration)
        if context_index is None:
            unscoped_kernel_count += 1
            unscoped_kernel_us += duration
            continue
        scoped_kernel_count += 1
        category = _kernel_category(str(event.get("name", "")))
        phase = str(contexts[context_index]["subphase"])
        totals[category][0] += duration
        totals[category][1] += 1
        phase_totals[phase][category][0] += duration
        phase_totals[phase][category][1] += 1
        context_totals[context_index][0] += duration
        context_totals[context_index][1] += 1
        name = str(event.get("name", ""))
        top_kernels[name][0] += duration
        top_kernels[name][1] += 1

    prefill_contexts = [row for row in contexts if row["phase"] == "prefill"]
    decode_contexts = [row for row in contexts if row["phase"] == "decode"]
    for index, row in enumerate(contexts):
        row["kernel_gpu_ms_sum"] = context_totals[index][0] / 1000.0
        row["kernel_calls"] = int(context_totals[index][1])
        row.pop("ts")
        row.pop("end")

    top_rows = [
        {
            "name": name,
            "gpu_ms_sum": values[0] / 1000.0,
            "calls": int(values[1]),
            "category": _kernel_category(name),
        }
        for name, values in sorted(top_kernels.items(), key=lambda item: -item[1][0])[
            :40
        ]
    ]
    return {
        "schema_version": 1,
        "trace": str(path),
        "classifier": "explicit kernel-name heuristic; inspect top_kernels",
        "duration_semantics": (
            "gpu_ms_sum is a sum of kernel durations and can exceed wall time "
            "when streams overlap"
        ),
        "trace_event_count": trace_event_count,
        "kernel_event_count": kernel_event_count,
        "scoped_kernel_count": scoped_kernel_count,
        "unscoped_kernel_count": unscoped_kernel_count,
        "unscoped_kernel_gpu_ms_sum": unscoped_kernel_us / 1000.0,
        "context_count": len(contexts),
        "prefill_context_count": len(prefill_contexts),
        "decode_context_count": len(decode_contexts),
        "prefill_q_tokens": sum(row["q_len"] for row in prefill_contexts),
        "decode_q_tokens": sum(row["q_len"] for row in decode_contexts),
        "contexts": contexts,
        "category_rows_scoped": _rows(totals),
        "phase_rows": {
            phase: _rows(phase_totals[phase])
            for phase in ("first_chunk", "continuation", "decode")
            if phase in phase_totals
        },
        "top_kernels": top_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_trace(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["phase_rows"], indent=2))


if __name__ == "__main__":
    main()
