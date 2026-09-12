# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the gfx1201 software-fused MXFP4 small-M linear kernel."""

import argparse
import gc
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class KernelConfig:
    block_n: int
    num_warps: int
    num_stages: int

    @property
    def name(self) -> str:
        return f"direct_n{self.block_n}_w{self.num_warps}_s{self.num_stages}"


def _launch_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    config: KernelConfig,
):
    return _mxfp4_small_m_kernel[
        (triton.cdiv(x.shape[0], 16), triton.cdiv(weight.shape[0], config.block_n))
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
        BLOCK_N=config.block_n,
        BLOCK_K=128,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )


def _run_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    config: KernelConfig,
) -> torch.Tensor:
    _launch_kernel(x, weight, scale, output, config)
    return output


def _measure_round_robin(
    operations: dict[str, Callable[[], torch.Tensor]],
    flush: torch.Tensor,
    warmups: int,
    samples: int,
) -> dict[str, list[float]]:
    for _ in range(warmups):
        for operation in operations.values():
            operation()
    torch.accelerator.synchronize()
    timings = {name: [] for name in operations}
    names = list(operations)
    for sample in range(samples):
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
    config: KernelConfig,
) -> None:
    output = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    compiled = _launch_kernel(x, weight, scale, output, config)
    torch.accelerator.synchronize()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = config.name.removeprefix("direct_")
    (artifact_dir / f"mxfp4_small_m_{stem}.s").write_text(compiled.asm["amdgcn"])
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
    metadata["config"] = {
        "block_m": 16,
        "block_n": config.block_n,
        "block_k": 128,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }
    (artifact_dir / f"mxfp4_small_m_{stem}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def _benchmark_shape(
    m: int,
    n: int,
    k: int,
    flush: torch.Tensor,
    warmups: int,
    samples: int,
    seed: int,
    configs: list[KernelConfig],
) -> dict:
    torch.manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.randint(120, 132, (n, k // 32), device="cuda", dtype=torch.uint8)
    dequantized = dequant_mxfp4(weight, scale, torch.bfloat16)
    reference = F.linear(x, dequantized)
    direct_outputs = {
        config.name: torch.empty((m, n), device="cuda", dtype=x.dtype)
        for config in configs
    }

    operations = {
        "weight_dequant": lambda: dequant_mxfp4(weight, scale, torch.bfloat16),
        "bf16_linear": lambda: F.linear(x, dequantized),
        "emulation": lambda: F.linear(x, dequant_mxfp4(weight, scale, torch.bfloat16)),
        "fused_wrapper": lambda: triton_mxfp4_small_m_linear(x, weight, scale),
    }
    for config in configs:
        output = direct_outputs[config.name]
        operations[config.name] = lambda output=output, config=config: _run_kernel(
            x, weight, scale, output, config
        )

    for name, operation in operations.items():
        candidate = operation()
        torch.accelerator.synchronize()
        if name == "weight_dequant":
            continue
        if not torch.equal(candidate, reference):
            delta = candidate.float() - reference.float()
            raise AssertionError(
                f"{name} output mismatch: max_abs={delta.abs().max().item()}, "
                f"rmse={delta.square().mean().sqrt().item()}"
            )

    raw_timing = _measure_round_robin(operations, flush, warmups, samples)
    timing = {name: _timing_summary(values) for name, values in raw_timing.items()}
    direct_speedup = {
        config.name: timing["emulation"]["median_us"] / timing[config.name]["median_us"]
        for config in configs
    }
    return {
        "m": m,
        "n": n,
        "k": k,
        "correctness": "bitwise_exact",
        "samples": samples,
        "flush_bytes": flush.numel() * flush.element_size(),
        "kernel_configs": [asdict(config) for config in configs],
        "direct_output_preallocated": True,
        "timing": timing,
        "direct_speedup_vs_emulation": direct_speedup,
        "wrapper_speedup_vs_emulation": timing["emulation"]["median_us"]
        / timing["fused_wrapper"]["median_us"],
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
    parser.add_argument("--block-n", type=int, nargs="+", default=[64])
    parser.add_argument("--num-warps", type=int, nargs="+", default=[4])
    parser.add_argument("--num-stages", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--shape-index",
        type=int,
        nargs="+",
        choices=range(len(MODEL_SHAPES)),
        default=range(len(MODEL_SHAPES)),
    )
    args = parser.parse_args()

    configs = [
        KernelConfig(block_n, num_warps, num_stages)
        for block_n in args.block_n
        for num_warps in args.num_warps
        for num_stages in args.num_stages
    ]
    selected_shapes = [
        (shape_index, MODEL_SHAPES[shape_index]) for shape_index in args.shape_index
    ]

    flush = torch.empty(args.flush_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)
    with args.output.open("x") as output:
        artifact_saved = False
        for shape_index, (n, k) in selected_shapes:
            for m in args.rows:
                result = _benchmark_shape(
                    m,
                    n,
                    k,
                    flush,
                    args.warmups,
                    args.samples,
                    args.seed + 10 * shape_index + m,
                    configs,
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
                    for config in configs:
                        _save_compiled_artifacts(
                            args.artifact_dir, x, weight, scale, config
                        )
                    artifact_saved = True
                gc.collect()
                torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
