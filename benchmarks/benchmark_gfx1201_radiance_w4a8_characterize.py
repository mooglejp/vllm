# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Characterize Radiance W4A8 kernels from saved profiler traces.

This is an offline, black-box analyzer.  It does not import Radiance source,
launch a model, or infer a launch grid that is not present in the trace.  The
report intentionally separates observations (kernel names, counts, and
durations) from clean-room design hypotheses that consume those observations.

The input traces are expected to contain ``execute_context_*`` GPU annotations
and ``kernel`` events.  Kernel durations are summed; they are not a wall-clock
critical path when streams overlap.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ijson
import regex as re

_CONTEXT = re.compile(r"^execute_context_([01])\((\d+)\)")


def _trace_events(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as trace_file:
        yield from ijson.items(trace_file, "traceEvents.item")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contexts(path: Path) -> tuple[list[dict[str, Any]], int]:
    contexts: list[dict[str, Any]] = []
    event_count = 0
    for event in _trace_events(path):
        event_count += 1
        if event.get("cat") != "gpu_user_annotation" or event.get("ph") != "X":
            continue
        match = _CONTEXT.match(str(event.get("name", "")))
        if match is None:
            continue
        start = float(event.get("ts", 0.0))
        duration = float(event.get("dur", 0.0))
        mode = match.group(1)
        contexts.append(
            {
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


def _kind(name: str) -> str | None:
    lowered = name.lower()
    if "radiance_mxfp4_fp8_gemm_folded" in lowered:
        return "native_folded"
    if "radiance_mxfp4_fp8_gemm_decode" in lowered:
        return "native_decode"
    if "dq_uint8_mxfp4_to_half" in lowered:
        return "fallback_dequant"
    if "qdq_mxfp4" in lowered:
        return "fallback_activation_qdq"
    if "cijk_" in lowered:
        return "fallback_cijk_gemm"
    return None


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "calls": 0,
            "sum_ms": 0.0,
            "mean_us": 0.0,
            "min_us": 0.0,
            "p50_us": 0.0,
            "p90_us": 0.0,
            "p95_us": 0.0,
            "p99_us": 0.0,
            "max_us": 0.0,
        }
    return {
        "calls": len(values),
        "sum_ms": sum(values) / 1000.0,
        "mean_us": sum(values) / len(values),
        "min_us": min(values),
        "p50_us": _percentile(values, 0.50),
        "p90_us": _percentile(values, 0.90),
        "p95_us": _percentile(values, 0.95),
        "p99_us": _percentile(values, 0.99),
        "max_us": max(values),
    }


def _phase_label(context: dict[str, Any]) -> str:
    return str(context["subphase"])


def characterize(path: Path) -> dict[str, Any]:
    contexts, trace_event_count = _contexts(path)
    if not contexts:
        raise ValueError(f"no execute_context annotations in {path}")
    starts = [float(row["ts"]) for row in contexts]

    selected: dict[str, list[float]] = defaultdict(list)
    by_phase: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_phase_q: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    by_context: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    kernel_names: Counter[str] = Counter()
    kernel_name_duration: defaultdict[str, float] = defaultdict(float)
    kernel_event_count = 0
    scoped_kernel_count = 0
    unscoped_kernel_count = 0

    for event in _trace_events(path):
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        kernel_event_count += 1
        name = str(event.get("name", ""))
        kind = _kind(name)
        if kind is None:
            continue
        duration = float(event.get("dur", 0.0))
        start = float(event.get("ts", 0.0))
        context_index = _containing_context(contexts, starts, start + 0.5 * duration)
        if context_index is None:
            unscoped_kernel_count += 1
            continue
        scoped_kernel_count += 1
        context = contexts[context_index]
        phase = _phase_label(context)
        selected[kind].append(duration)
        by_phase[phase][kind].append(duration)
        by_phase_q[phase][int(context["q_len"])][kind].append(duration)
        by_context[context_index][kind].append(duration)
        kernel_names[name] += 1
        kernel_name_duration[name] += duration

    context_rows: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        row = {
            "ordinal": context["ordinal"],
            "phase": context["phase"],
            "subphase": context["subphase"],
            "q_len": context["q_len"],
            "wall_ms": context["duration_us"] / 1000.0,
            "selected": {
                kind: _distribution(by_context[index].get(kind, []))
                for kind in sorted(by_context[index])
            },
        }
        context_rows.append(row)

    phase_rows: dict[str, Any] = {}
    for phase, q_rows in by_phase_q.items():
        phase_rows[phase] = {
            "all_q_lens": {
                kind: _distribution(values) for kind, values in by_phase[phase].items()
            },
            "by_q_len": {
                str(q_len): {
                    kind: _distribution(values)
                    for kind, values in sorted(kind_rows.items())
                }
                for q_len, kind_rows in sorted(q_rows.items())
            },
        }

    top_names = [
        {
            "name": name,
            "calls": calls,
            "sum_ms": kernel_name_duration[name] / 1000.0,
            "kind": _kind(name),
        }
        for name, calls in kernel_names.most_common()
    ]

    return {
        "schema_version": 1,
        "trace": str(path),
        "trace_sha256": _sha256(path),
        "trace_size_bytes": path.stat().st_size,
        "trace_event_count": trace_event_count,
        "kernel_event_count": kernel_event_count,
        "selected_scoped_kernel_count": scoped_kernel_count,
        "selected_unscoped_kernel_count": unscoped_kernel_count,
        "context_count": len(contexts),
        "prefill_context_count": sum(
            context["phase"] == "prefill" for context in contexts
        ),
        "prefill_q_tokens": sum(
            context["q_len"] for context in contexts if context["phase"] == "prefill"
        ),
        "selected_distributions": {
            kind: _distribution(values) for kind, values in sorted(selected.items())
        },
        "phase_rows": phase_rows,
        "context_rows": context_rows,
        "selected_kernel_names": top_names,
        "limitations": [
            "Durations are summed kernel durations, not a wall-clock critical path.",
            "The trace exposes no grid, block, wave-count, VGPR, LDS, or ISA metadata.",
            "Kernel signature template parameters and pointer types are observations; "
            "their semantic argument names are not inferred.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = characterize(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["phase_rows"], indent=2))


if __name__ == "__main__":
    main()
