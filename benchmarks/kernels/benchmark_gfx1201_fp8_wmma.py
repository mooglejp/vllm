# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate raw gfx1201 FP8-WMMA throughput from the P3 W4A8 candidate.

This is a diagnostic-only benchmark.  Both inputs are pre-expanded FP8 byte
matrices, no MXFP4 decode or E8M0 scale is performed, and the extension is
loaded out of tree rather than registered with vLLM.  Allocation, input
generation, compilation, and correctness checks are outside the timed region.
"""

import argparse
import json
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

MODEL_SHAPES = (
    (5120, 6144),
    (34816, 5120),
    (5120, 17408),
    (16384, 5120),
    (96, 5120),
    (14336, 5120),
)
DEFAULT_ROWS = (256,)
FP8_DTYPE = torch.float8_e4m3fn


@dataclass(frozen=True)
class TimingConfig:
    warmups: int = 5
    samples: int = 20


def _git_revision() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository}",
                "-C",
                str(repository),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _load_extension(build_directory: Path, verbose: bool) -> Any:
    from torch.utils.cpp_extension import load

    build_directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("gfx1201_fp8_wmma_microbenchmark.cu")
    return load(
        name="gfx1201_fp8_wmma_microbenchmark",
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--offload-arch=gfx1201"],
        with_cuda=True,
        verbose=verbose,
    )


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


def _fp8_bytes(shape: tuple[int, int], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    # Keep values finite and moderate so the FP64 oracle remains informative;
    # the benchmark's source of truth is the resulting raw FP8 byte tensor.
    values = torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    values = (values.float() * 0.25).clamp(-4.0, 4.0)
    return values.to(FP8_DTYPE).view(torch.uint8)


def _oracle_slice(
    a: torch.Tensor, b: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    a64 = a[:rows].view(FP8_DTYPE).double()
    b64 = b[:cols].view(FP8_DTYPE).double()
    return torch.mm(a64, b64.t())


def _error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = actual.double() - reference
    return {
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
    }


def _check_correctness(
    extension: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
) -> dict[str, object]:
    if output.numel() <= 1_000_000:
        rows, cols = a.size(0), b.size(0)
        scope = "full output"
    else:
        rows = min(a.size(0), 16)
        cols = min(b.size(0), 32)
        scope = "first output slice"
    extension.fp8_wmma_gemm(a, b, output)
    fp32_actual = torch.mm(
        a[:rows].view(FP8_DTYPE).float(),
        b[:cols].view(FP8_DTYPE).float().t(),
    )
    torch.accelerator.synchronize()
    oracle = _oracle_slice(a, b, rows, cols)
    actual = output[:rows, :cols]
    fp32_error = _error(actual, fp32_actual.double())
    oracle_error = _error(actual, oracle)
    return {
        "oracle": "FP64 decode of the exact input FP8 bytes, accumulation in FP64",
        "scope": scope,
        "slice": [rows, cols],
        "fp64": oracle_error,
        "torch_fp32": fp32_error,
        "finite": bool(torch.isfinite(actual).all()),
        "tolerance": {
            "max_abs": 0.25,
            "rmse": 0.01,
            "purpose": "diagnostic sanity check, not a P3 adoption gate",
        },
        "passed": bool(
            torch.isfinite(actual).all()
            and oracle_error["max_abs"] <= 0.25
            and oracle_error["rmse"] <= 0.01
        ),
    }


def _tail_correctness_probe(extension: Any) -> dict[str, object]:
    """Check the non-tile M/N tails with a compact FP64 oracle."""

    m, n, k = 129, 17, 64
    a = _fp8_bytes((m, k), 1201001)
    b = _fp8_bytes((n, k), 1201002)
    output = torch.empty((m, n), device="cuda", dtype=torch.float32)
    extension.fp8_wmma_gemm(a, b, output)
    torch.accelerator.synchronize()
    error = _error(output, _oracle_slice(a, b, m, n))
    return {
        "shape": [m, n, k],
        "oracle": "FP64 decode of the exact input FP8 bytes, full output",
        "error": error,
        "finite": bool(torch.isfinite(output).all()),
        "tolerance": {"max_abs": 0.25, "rmse": 0.01},
        "passed": bool(
            torch.isfinite(output).all()
            and error["max_abs"] <= 0.25
            and error["rmse"] <= 0.01
        ),
    }


def _benchmark_case(
    extension: Any,
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
    a_fp32 = a.view(FP8_DTYPE).float()
    b_fp32 = b.view(FP8_DTYPE).float()
    a_bf16 = a_fp32.to(torch.bfloat16)
    b_bf16 = b_fp32.to(torch.bfloat16)
    output = torch.empty((m, n), device="cuda", dtype=torch.float32)
    fp32_output = torch.empty_like(output)
    bf16_output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    def fp8_wmma_fp32() -> None:
        extension.fp8_wmma_gemm(a, b, output)

    def torch_fp32_mm() -> None:
        torch.mm(a_fp32, b_fp32.t(), out=fp32_output)

    def torch_bf16_mm() -> None:
        torch.mm(a_bf16, b_bf16.t(), out=bf16_output)

    correctness = (
        {"skipped": True}
        if skip_correctness
        else _check_correctness(extension, a, b, output)
    )
    torch.accelerator.synchronize()
    operations = {
        "fp8_wmma_fp32": fp8_wmma_fp32,
        "torch_fp32_mm": torch_fp32_mm,
        "torch_bf16_mm": torch_bf16_mm,
    }
    raw = _measure_round_robin(operations, flush, config)
    timing = {name: _summary(values) for name, values in raw.items()}
    flop = 2.0 * m * n * k
    throughput = {
        name: flop / (values["median_us"] * 1e-6) / 1e12
        for name, values in timing.items()
    }
    return {
        "m": m,
        "n": n,
        "k": k,
        "seed": seed,
        "timing": timing,
        "effective_tflops": throughput,
        "correctness": correctness,
        "inputs": {
            "a": "raw FP8 E4M3 bytes, prequantized",
            "b": "raw FP8 E4M3 bytes, preexpanded",
            "scales": "none",
            "mxfp4_decode": False,
        },
        "timed_region": (
            "one fixed-buffer GEMM call; allocation, conversion, compilation, "
            "oracle, and input generation are excluded"
        ),
    }


def _metadata(args: argparse.Namespace) -> dict[str, object]:
    properties = torch.cuda.get_device_properties()
    return {
        "revision": _git_revision(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(torch.accelerator.current_accelerator()),
        "device_name": properties.name or properties.gcnArchName,
        "gcn_arch": properties.gcnArchName,
        "timing_config": asdict(TimingConfig(args.warmups, args.samples)),
        "rows": args.rows,
        "shape_indices": args.shape_index,
        "flush_mib": args.flush_mib,
        "build_directory": str(args.build_directory),
        "source": "benchmarks/kernels/gfx1201_fp8_wmma_microbenchmark.cu",
        "build": {
            "torch_cpp_extension_load": True,
            "with_cuda": True,
            "extra_cflags": ["-O2"],
            "extra_cuda_cflags": ["-O2", "--offload-arch=gfx1201"],
        },
        "scope": (
            "diagnostic only; no production registration, dispatch, threshold, "
            "workspace reuse, or P4 work"
        ),
        "operation": "preexpanded FP8 E4M3 A[M,K] x B[N,K]^T -> FP32 C[M,N]",
        "baseline": (
            "preexpanded FP32 and BF16 torch.mm, separately timed; neither "
            "includes FP8 conversion"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape-index", type=int, nargs="+", default=None)
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument(
        "--build-directory",
        type=Path,
        default=Path("/tmp/tq-gfx1201-fp8-wmma-build"),
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

    extension = _load_extension(args.build_directory, args.verbose_build)
    config = TimingConfig(args.warmups, args.samples)
    flush = torch.empty(args.flush_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)
    tail_correctness = _tail_correctness_probe(extension)
    if not tail_correctness["passed"]:
        raise RuntimeError(f"FP8 WMMA tail correctness failed: {tail_correctness}")
    selected = (
        range(len(MODEL_SHAPES)) if args.shape_index is None else args.shape_index
    )
    for index in selected:
        if index < 0 or index >= len(MODEL_SHAPES):
            raise ValueError(f"shape index {index} is out of range")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        metadata = _metadata(args)
        metadata["tail_correctness"] = tail_correctness
        output_file.write(json.dumps({"metadata": metadata}) + "\n")
        output_file.flush()
        print(json.dumps({"metadata": metadata}), flush=True)
        for shape_index in selected:
            n, k = MODEL_SHAPES[shape_index]
            for m in args.rows:
                result = _benchmark_case(
                    extension,
                    m,
                    n,
                    k,
                    flush,
                    config,
                    seed=1201 + shape_index * 10000 + m,
                    skip_correctness=args.skip_correctness,
                )
                output_file.write(json.dumps(result) + "\n")
                output_file.flush()
                print(json.dumps(result), flush=True)
        print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
