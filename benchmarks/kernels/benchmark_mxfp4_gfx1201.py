# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the gfx1201 software-fused MXFP4 small-M linear kernel."""

import argparse
import gc
import json
import random
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.mxfp4.triton_gfx1201 import (
    _mxfp4_small_m_kernel,
    triton_mxfp4_small_m_linear,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
)
from vllm.triton_utils import triton

MODEL_SHAPES = (
    (5120, 6144),
    (34816, 5120),
    (5120, 17408),
    (16384, 5120),
    (96, 5120),
    (14336, 5120),
)


def _measure(
    operation: Callable[[], torch.Tensor],
    flush: torch.Tensor,
    warmups: int,
    samples: int,
) -> list[float]:
    for _ in range(warmups):
        operation()
    torch.accelerator.synchronize()
    timings = []
    for _ in range(samples):
        flush.zero_()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000)
    return timings


def _timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def _save_compiled_artifacts(
    artifact_dir: Path,
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    output = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    compiled = _mxfp4_small_m_kernel[
        (triton.cdiv(x.shape[0], 16), triton.cdiv(weight.shape[0], 32))
    ](
        x,
        weight,
        scale,
        output,
        x.shape[0],
        weight.shape[0],
        x.shape[1],
        *x.stride(),
        *weight.stride(),
        *scale.stride(),
        *output.stride(),
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=128,
        num_warps=2,
        num_stages=1,
    )
    torch.accelerator.synchronize()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "mxfp4_small_m.s").write_text(compiled.asm["amdgcn"])
    metadata = {
        name: getattr(compiled.metadata, name)
        for name in (
            "name",
            "num_warps",
            "num_stages",
            "shared",
            "n_regs",
            "n_spills",
        )
        if hasattr(compiled.metadata, name)
    }
    (artifact_dir / "mxfp4_small_m.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def _benchmark_shape(
    m: int,
    n: int,
    k: int,
    flush: torch.Tensor,
    warmups: int,
    samples: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.randint(120, 132, (n, k // 32), device="cuda", dtype=torch.uint8)
    dequantized = dequant_mxfp4(weight, scale, torch.bfloat16)
    reference = F.linear(x, dequantized)
    candidate = triton_mxfp4_small_m_linear(x, weight, scale)
    torch.accelerator.synchronize()
    if not torch.equal(candidate, reference):
        delta = candidate.float() - reference.float()
        raise AssertionError(
            f"output mismatch: max_abs={delta.abs().max().item()}, "
            f"rmse={delta.square().mean().sqrt().item()}"
        )

    operations = {
        "weight_dequant": lambda: dequant_mxfp4(weight, scale, torch.bfloat16),
        "bf16_linear": lambda: F.linear(x, dequantized),
        "emulation": lambda: F.linear(x, dequant_mxfp4(weight, scale, torch.bfloat16)),
        "fused": lambda: triton_mxfp4_small_m_linear(x, weight, scale),
    }
    names = list(operations)
    random.Random(seed).shuffle(names)
    timing = {
        name: _timing_summary(_measure(operations[name], flush, warmups, samples))
        for name in names
    }
    return {
        "m": m,
        "n": n,
        "k": k,
        "correctness": "bitwise_exact",
        "samples": samples,
        "flush_bytes": flush.numel() * flush.element_size(),
        "timing": timing,
        "speedup_vs_emulation": (
            timing["emulation"]["median_us"] / timing["fused"]["median_us"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()

    flush = torch.empty(args.flush_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)
    with args.output.open("x") as output:
        artifact_saved = False
        for shape_index, (n, k) in enumerate(MODEL_SHAPES):
            for m in args.rows:
                result = _benchmark_shape(
                    m,
                    n,
                    k,
                    flush,
                    args.warmups,
                    args.samples,
                    args.seed + 10 * shape_index + m,
                )
                encoded = json.dumps(result, sort_keys=True)
                output.write(encoded + "\n")
                output.flush()
                print(encoded, flush=True)
                if args.artifact_dir is not None and not artifact_saved:
                    torch.manual_seed(args.seed)
                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                    weight = torch.randint(
                        0, 256, (n, k // 2), device="cuda", dtype=torch.uint8
                    )
                    scale = torch.randint(
                        120,
                        132,
                        (n, k // 32),
                        device="cuda",
                        dtype=torch.uint8,
                    )
                    _save_compiled_artifacts(args.artifact_dir, x, weight, scale)
                    artifact_saved = True
                gc.collect()
                torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
