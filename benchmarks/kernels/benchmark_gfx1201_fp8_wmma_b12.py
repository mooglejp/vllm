# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the fixed-A3 B-load/LDS-layout diagnostic A/B variants.

This benchmark is deliberately narrower than P3-v2a. It keeps the existing
4-wave 64x128 A3 mapping and compares only two B staging variants:
B1 changes the global-load assignment order, while B2 also changes the
physical LDS order and uses a col-major matrix-B fragment. Inputs are raw
pre-expanded FP8 bytes; there is no MXFP4 decode, scale, quantization,
production registration, or model integration.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from benchmarks.kernels.benchmark_gfx1201_fp8_wmma import (
        FP8_DTYPE,
        MODEL_SHAPES,
        _error,
        _fp8_bytes,
        _git_revision,
        _load_extension,
        _oracle_slice,
    )
except ModuleNotFoundError:
    from benchmark_gfx1201_fp8_wmma import (  # type: ignore[no-redef]
        FP8_DTYPE,
        MODEL_SHAPES,
        _error,
        _fp8_bytes,
        _git_revision,
        _load_extension,
        _oracle_slice,
    )


BASE_REVISION = "fba76b224a3895dc920c0756076600688c51c4d2"
VARIANTS = (
    ("old_a3", "fp8_wmma_gemm_4wave_64x128"),
    ("b1_k_priority", "fp8_wmma_gemm_4wave_64x128_b1"),
    ("b2_lds_nk_col_major", "fp8_wmma_gemm_4wave_64x128_b2"),
)
VARIANT_NAMES = tuple(name for name, _ in VARIANTS)
PRODUCTION_CALL_WEIGHTS = (64, 64, 64, 48, 48, 16)
LARGE_SHAPE_INDICES = tuple(
    index for index, (n, _k) in enumerate(MODEL_SHAPES) if n >= 512
)
assert len(MODEL_SHAPES) == len(PRODUCTION_CALL_WEIGHTS)


@dataclass(frozen=True)
class TimingConfig:
    warmups: int = 5
    samples: int = 20


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def _measure_round_robin(
    operations: dict[str, Callable[[], object]],
    flush: torch.Tensor,
    config: TimingConfig,
) -> dict[str, list[float]]:
    for _ in range(config.warmups):
        for operation in operations.values():
            operation()
    torch.accelerator.synchronize()
    timings = {name: [] for name in operations}
    names = list(operations)
    for sample in range(config.samples):
        offset = sample % len(names)
        for name in names[offset:] + names[:offset]:
            flush.zero_()
            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)
            start.record()
            operations[name]()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) * 1000)
    return timings


def _invoke(
    extension: Any,
    method: str,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
) -> None:
    getattr(extension, method)(a, b, output)


def _representative_indices(size: int, boundaries: tuple[int, ...]) -> list[int]:
    values = {0, 1, size - 2, size - 1}
    values.update(value for value in boundaries if 0 <= value < size)
    return sorted(value for value in values if 0 <= value < size)


def _correctness_case(
    extension: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    case: str,
    oracle_rows: list[int] | None = None,
    oracle_cols: list[int] | None = None,
) -> dict[str, object]:
    for name, method in VARIANTS:
        _invoke(extension, method, a, b, outputs[name])
    torch.accelerator.synchronize()

    if oracle_rows is None or oracle_cols is None:
        rows = list(range(a.size(0)))
        cols = list(range(b.size(0)))
        scope = "full output"
    else:
        rows = oracle_rows
        cols = oracle_cols
        scope = "representative rows and columns"

    row_index = torch.tensor(rows, device=a.device, dtype=torch.long)
    col_index = torch.tensor(cols, device=a.device, dtype=torch.long)
    oracle = _oracle_slice(
        a.index_select(0, row_index),
        b.index_select(0, col_index),
        len(rows),
        len(cols),
    )
    baseline = outputs["old_a3"]
    candidates: dict[str, object] = {}
    all_passed = True
    for name in VARIANT_NAMES:
        actual = outputs[name].index_select(0, row_index).index_select(1, col_index)
        fp64_error = _error(actual, oracle)
        bitwise_vs_old = bool(torch.equal(outputs[name], baseline))
        finite = bool(torch.isfinite(actual).all())
        diagnostic_passed = bool(
            finite and fp64_error["max_abs"] <= 0.25 and fp64_error["rmse"] <= 0.01
        )
        candidate_passed = bitwise_vs_old and diagnostic_passed
        all_passed = all_passed and candidate_passed
        candidates[name] = {
            "finite": finite,
            "bitwise_vs_old_a3": bitwise_vs_old,
            "fp64": fp64_error,
            "passed": candidate_passed,
        }
    return {
        "case": case,
        "shape": [a.size(0), b.size(0), a.size(1)],
        "oracle": "FP64 decode of exact input FP8 bytes with FP64 accumulation",
        "scope": scope,
        "oracle_rows": rows,
        "oracle_cols": cols,
        "diagnostic_tolerance": {
            "max_abs": 0.25,
            "rmse": 0.01,
            "purpose": (
                "existing pure-FP8 diagnostic envelope; no P2.2 attention "
                "threshold is reused"
            ),
        },
        "candidates": candidates,
        "passed": all_passed,
    }


def _column_distinct_inputs(
    m: int, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    a_values = torch.ones((m, k), device="cuda", dtype=torch.float32)
    column_values = torch.linspace(-4.0, 4.0, n, device="cuda")
    b_values = column_values[:, None].expand(n, k).contiguous()
    return (
        a_values.to(FP8_DTYPE).view(torch.uint8),
        b_values.to(FP8_DTYPE).view(torch.uint8),
    )


def _small_correctness_cases() -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    yield (
        "random_tail_129x17x64",
        _fp8_bytes((129, 64), 1202001),
        _fp8_bytes((17, 64), 1202002),
    )
    distinct_a, distinct_b = _column_distinct_inputs(129, 129, 64)
    yield "column_distinct_129x129x64", distinct_a, distinct_b
    yield (
        "random_full_129x129x64",
        _fp8_bytes((129, 64), 1202003),
        _fp8_bytes((129, 64), 1202004),
    )
    yield (
        "random_all_tails_65x129x65",
        _fp8_bytes((65, 65), 1202005),
        _fp8_bytes((129, 65), 1202006),
    )


def _run_small_correctness(extension: Any) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for case, a, b in _small_correctness_cases():
        outputs = {
            name: torch.empty(
                (a.size(0), b.size(0)), device="cuda", dtype=torch.float32
            )
            for name in VARIANT_NAMES
        }
        results.append(_correctness_case(extension, a, b, outputs, case))
    return results


def _large_correctness(
    extension: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    shape_index: int,
    m: int,
    n: int,
    k: int,
) -> dict[str, object]:
    row_boundaries = (16, 32, 48, 64, 96, 128, 192, 256)
    col_boundaries = (16, 32, 64, 96, 128, 256, 512)
    rows = _representative_indices(m, row_boundaries)
    cols = _representative_indices(n, col_boundaries)
    result = _correctness_case(
        extension,
        a,
        b,
        outputs,
        f"shape_{shape_index}_m{m}_n{n}_k{k}",
        rows,
        cols,
    )
    result["full_output_bitwise_vs_old_a3"] = {
        name: bool(torch.equal(outputs[name], outputs["old_a3"]))
        for name in VARIANT_NAMES
    }
    result["full_output_finite"] = {
        name: bool(torch.isfinite(outputs[name]).all()) for name in VARIANT_NAMES
    }
    result["full_output_elements"] = int(outputs["old_a3"].numel())
    result["shape_index"] = shape_index
    result["m"] = m
    result["passed"] = bool(
        result["passed"]
        and all(result["full_output_bitwise_vs_old_a3"].values())
        and all(result["full_output_finite"].values())
    )
    return result


def _benchmark_case(
    extension: Any,
    shape_index: int,
    m: int,
    n: int,
    k: int,
    flush: torch.Tensor,
    config: TimingConfig,
    seed: int,
    skip_correctness: bool,
) -> dict[str, object]:
    a = _fp8_bytes((m, k), seed)
    b = _fp8_bytes((n, k), seed + 1)
    outputs = {
        name: torch.empty((m, n), device="cuda", dtype=torch.float32)
        for name in VARIANT_NAMES
    }
    bf16_a = a.view(FP8_DTYPE).to(torch.bfloat16)
    bf16_b_t = b.view(FP8_DTYPE).to(torch.bfloat16).t()
    bf16_output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    if skip_correctness:
        correctness: dict[str, object] = {"skipped": True}
    else:
        correctness = _large_correctness(extension, a, b, outputs, shape_index, m, n, k)
    torch.accelerator.synchronize()

    operations: dict[str, Callable[[], object]] = {
        name: lambda method=method, output=outputs[name]: _invoke(
            extension, method, a, b, output
        )
        for name, method in VARIANTS
    }
    operations["torch_bf16_mm"] = lambda: torch.mm(bf16_a, bf16_b_t, out=bf16_output)
    raw = _measure_round_robin(operations, flush, config)
    timing = {name: _summary(values) for name, values in raw.items()}
    flop = 2.0 * m * n * k
    effective_tflops = {
        name: flop / (float(values["median_us"]) * 1.0e6)
        for name, values in timing.items()
    }
    old_us = float(timing["old_a3"]["median_us"])
    speedup = {
        name: old_us / float(values["median_us"])
        for name, values in timing.items()
        if name != "old_a3"
    }
    bf16_us = float(timing["torch_bf16_mm"]["median_us"])
    ratio_to_bf16 = {
        name: float(values["median_us"]) / bf16_us for name, values in timing.items()
    }
    return {
        "shape_index": shape_index,
        "m": m,
        "n": n,
        "k": k,
        "seed": seed,
        "shape_call_weight": PRODUCTION_CALL_WEIGHTS[shape_index],
        "timing": timing,
        "effective_tflops": effective_tflops,
        "speedup_over_old_a3": speedup,
        "time_ratio_to_bf16_mm": ratio_to_bf16,
        "correctness": correctness,
        "inputs": {
            "a": "raw FP8 E4M3 bytes, prequantized",
            "b": "raw FP8 E4M3 bytes, preexpanded",
            "b_shape": [n, k],
            "b_stride": [k, 1],
            "scales": "none",
            "mxfp4_decode": False,
        },
        "timed_region": (
            "one fixed-buffer A3/B1/B2 or BF16 torch.mm call; allocation, "
            "conversion, compilation, correctness, and input generation are "
            "excluded"
        ),
    }


def _candidate_correctness_passed(
    record: dict[str, object], name: str, requested: bool
) -> bool:
    if not requested:
        return False
    correctness = record.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("skipped"):
        return False
    candidates = correctness.get("candidates")
    if not isinstance(candidates, dict):
        return False
    candidate = candidates.get(name)
    return isinstance(candidate, dict) and bool(candidate.get("passed"))


def _weighted_summary(
    records: list[dict[str, object]],
    small_correctness: list[dict[str, object]] | dict[str, object],
    correctness_requested: bool,
) -> dict[str, object]:
    by_m: dict[str, dict[str, object]] = {}
    expected = {(shape_index, 256) for shape_index in LARGE_SHAPE_INDICES}
    present = {
        (int(record["shape_index"]), int(record["m"]))
        for record in records
        if int(record["n"]) >= 512 and int(record["m"]) == 256
    }
    complete = expected.issubset(present)
    for record in records:
        shape_index = int(record["shape_index"])
        if int(record["n"]) < 512:
            continue
        m = str(record["m"])
        weight = float(PRODUCTION_CALL_WEIGHTS[shape_index])
        timing = record["timing"]
        assert isinstance(timing, dict)
        bucket = by_m.setdefault(
            m,
            {
                "total_calls": 0.0,
                **{f"{name}_us": 0.0 for name in VARIANT_NAMES},
                "torch_bf16_mm_us": 0.0,
            },
        )
        bucket["total_calls"] = float(bucket["total_calls"]) + weight
        for name in VARIANT_NAMES:
            bucket[f"{name}_us"] = float(bucket[f"{name}_us"]) + weight * float(
                timing[name]["median_us"]
            )
        bucket["torch_bf16_mm_us"] = float(bucket["torch_bf16_mm_us"]) + weight * float(
            timing["torch_bf16_mm"]["median_us"]
        )

    for bucket in by_m.values():
        old_us = float(bucket["old_a3_us"])
        bf16_us = float(bucket["torch_bf16_mm_us"])
        bucket["speedup_over_old_a3"] = {
            name: old_us / float(bucket[f"{name}_us"]) for name in VARIANT_NAMES
        }
        bucket["time_ratio_to_bf16_mm"] = {
            name: float(bucket[f"{name}_us"]) / bf16_us for name in VARIANT_NAMES
        }

    small_passed = (
        correctness_requested
        and isinstance(small_correctness, list)
        and bool(small_correctness)
        and all(bool(case.get("passed")) for case in small_correctness)
    )
    gates: dict[str, object] = {}
    m256 = by_m.get("256", {})
    for name in VARIANT_NAMES[1:]:
        case_correct = correctness_requested and complete
        if case_correct:
            for record in records:
                if int(record["n"]) >= 512 and int(record["m"]) == 256:
                    case_correct = case_correct and _candidate_correctness_passed(
                        record, name, correctness_requested
                    )
        speedup = None
        speedups = m256.get("speedup_over_old_a3")
        if isinstance(speedups, dict):
            speedup = speedups.get(name)
        speed_passed = isinstance(speedup, (float, int)) and float(speedup) >= 1.25
        gates[name] = {
            "m": 256,
            "n_at_least": 512,
            "weighted_speedup_over_old_a3": speedup,
            "threshold": 1.25,
            "weighted_cases_complete": complete,
            "small_correctness_passed": small_passed,
            "large_correctness_passed": case_correct,
            "correctness_required": True,
            "passed": bool(complete and small_passed and case_correct and speed_passed),
        }
    return {
        "shape_call_weights": [
            {
                "shape_index": index,
                "n": MODEL_SHAPES[index][0],
                "k": MODEL_SHAPES[index][1],
                "calls_per_target_forward": PRODUCTION_CALL_WEIGHTS[index],
            }
            for index in range(len(MODEL_SHAPES))
        ],
        "weighted_shape_indices": list(LARGE_SHAPE_INDICES),
        "by_m": by_m,
        "m256_expected_cases": [list(case) for case in sorted(expected)],
        "m256_present_cases": [list(case) for case in sorted(present)],
        "m256_missing_cases": [list(case) for case in sorted(expected - present)],
        "diagnostic_gate": {
            "baseline": "old_a3",
            "threshold": 1.25,
            "candidate_scope": "M=256, N>=512, call-weighted median time",
            "correctness_required": True,
            "candidate_results": gates,
            "note": (
                "This is a B-load/layout diagnostic criterion, not a W4A8 "
                "adoption gate; M=64 is reported independently."
            ),
        },
    }


def _metadata(
    args: argparse.Namespace,
    config: TimingConfig,
    selected: list[int],
    extension: Any,
) -> dict[str, object]:
    properties = torch.cuda.get_device_properties()
    return {
        "base_revision": BASE_REVISION,
        "revision": _git_revision(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(torch.accelerator.current_accelerator()),
        "device_name": properties.name or properties.gcnArchName,
        "gcn_arch": properties.gcnArchName,
        "timing_config": asdict(config),
        "rows": args.rows,
        "shape_indices": selected,
        "flush_mib": args.flush_mib,
        "build_directory": str(args.build_directory),
        "extension_file": str(getattr(extension, "__file__", "unknown")),
        "source": "benchmarks/kernels/gfx1201_fp8_wmma_microbenchmark.cu",
        "phase": "P3-v2 fixed-A3 B-load/LDS-layout limited A/B",
        "variants": [
            {
                "name": name,
                "method": method,
                "tile": "4-wave 64x128",
                "global_b_order": ("K-priority" if name != "old_a3" else "A3 original"),
                "lds_layout": (
                    "[N,K] physical / col_major"
                    if name == "b2_lds_nk_col_major"
                    else "logical [K,N] row_major"
                ),
            }
            for name, method in VARIANTS
        ],
        "bf16_control": (
            "same FP8 bytes converted to BF16 before timing; BF16 output and "
            "FP32 candidate output are a speed diagnostic, not a shared "
            "precision contract"
        ),
        "unchanged_contract": [
            "A loading",
            "K=16 WMMA order",
            "accumulator and FP32 output",
            "synchronization locations",
            "global B byte shape/stride",
            "no MXFP4 decode, scale, quantization, or production integration",
        ],
        "production_call_weights_source": (
            "docs/design/turboquant_gfx1201_mxfp4_launch_tuning.md"
        ),
        "diagnostic_gate": {
            "m": 256,
            "n_at_least": 512,
            "weighted_speedup_over_old_a3": 1.25,
            "correctness_required": True,
        },
        "command": " ".join(["benchmark_gfx1201_fp8_wmma_b12.py", *sys.argv[1:]]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape-index", type=int, nargs="+", default=None)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument(
        "--build-directory",
        type=Path,
        default=Path("/tmp/tq-gfx1201-p3-v2-b12-build-714"),
    )
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    if not torch.accelerator.is_available() or torch.version.hip is None:
        raise RuntimeError("This benchmark requires a ROCm GPU")
    if any(row < 1 for row in args.rows):
        raise ValueError("rows must be positive")
    if args.warmups < 0 or args.samples < 1:
        raise ValueError("warmups must be nonnegative and samples must be positive")
    selected = list(
        range(len(MODEL_SHAPES)) if args.shape_index is None else args.shape_index
    )
    for index in selected:
        if index < 0 or index >= len(MODEL_SHAPES):
            raise ValueError(f"shape index {index} is out of range")

    extension = _load_extension(args.build_directory, args.verbose_build)
    config = TimingConfig(args.warmups, args.samples)
    flush = torch.empty(args.flush_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)
    if args.skip_correctness:
        small_correctness: list[dict[str, object]] | dict[str, object] = {
            "skipped": True
        }
    else:
        small_correctness = _run_small_correctness(extension)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with args.output.open("w") as output:
        metadata = _metadata(args, config, selected, extension)
        metadata["small_correctness"] = small_correctness
        output.write(json.dumps({"metadata": metadata}) + "\n")
        output.flush()
        print(json.dumps({"metadata": metadata}), flush=True)
        for shape_index in selected:
            n, k = MODEL_SHAPES[shape_index]
            for m in args.rows:
                record = _benchmark_case(
                    extension,
                    shape_index,
                    m,
                    n,
                    k,
                    flush,
                    config,
                    1202000 + shape_index * 10000 + m,
                    args.skip_correctness,
                )
                records.append(record)
                output.write(json.dumps(record) + "\n")
                output.flush()
                print(json.dumps(record), flush=True)
        weighted = _weighted_summary(
            records, small_correctness, not args.skip_correctness
        )
        output.write(json.dumps({"weighted_summary": weighted}) + "\n")
        output.flush()
        print(json.dumps({"weighted_summary": weighted}), flush=True)
        decision = {
            "stop_after_fixed_ab": True,
            "production_integration": False,
            "mx_fp4_fusion": False,
            "additional_tile_search": False,
            "p3_v2b_authorized": False,
            "note": (
                "This artifact only determines whether the fixed-A3 B-load/LDS "
                "change merits a later hypothesis; it does not reopen v2b."
            ),
        }
        output.write(json.dumps({"decision": decision}) + "\n")
        output.flush()
        print(json.dumps({"decision": decision}), flush=True)
        print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
