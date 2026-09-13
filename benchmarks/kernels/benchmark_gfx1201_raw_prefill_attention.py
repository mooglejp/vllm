# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a narrow raw first-chunk attention candidate on gfx1201.

This is a P4.1 benchmark-only experiment.  It does not register an operator
or alter the TurboQuant production dispatcher.  The candidate is a clean-room
online-softmax Triton kernel for the fixed first-chunk contract
``[M, Hq=24, D=256]`` query, ``[M, Hk=4, D=256]`` raw K/V, causal BF16 GQA6.

The current runnable backend in the P4.0 lane is explicit Math SDPA because
the available gfx1201 CK flash kernel segfaults for this shape.  Both paths
receive the same tensors and preallocated destination.  Math SDPA still has
backend-internal temporaries, which are intentionally retained in its timing
because they are part of the current fallback cost.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vllm.triton_utils import triton

H_Q = 24
H_K = 4
HEAD_DIM = 256
GROUP_SIZE = H_Q // H_K
LOG2E = 1.4426950408889634


@triton.jit
def _raw_causal_gqa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qm,
    stride_qh,
    stride_km,
    stride_kh,
    stride_vm,
    stride_vh,
    stride_om,
    stride_oh,
    num_tokens,
    scale,
    GROUP_SIZE_: triton.language.constexpr,
    LOG2E_: triton.language.constexpr,
    BLOCK_M: triton.language.constexpr,
    BLOCK_N: triton.language.constexpr,
    HEAD_DIM_: triton.language.constexpr,
):
    """One program computes a small query tile for one GQA query head."""

    pid_m = triton.language.program_id(0)
    q_head = triton.language.program_id(1)
    q_rows = pid_m * BLOCK_M + triton.language.arange(0, BLOCK_M)
    d = triton.language.arange(0, HEAD_DIM_)
    q_mask = q_rows < num_tokens
    d_mask = d < HEAD_DIM_

    q = triton.language.load(
        q_ptr + q_rows[:, None] * stride_qm + q_head * stride_qh + d[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    q = q.to(triton.language.float32)
    kv_head = q_head // GROUP_SIZE_
    m_i = triton.language.full((BLOCK_M,), -float("inf"), triton.language.float32)
    l_i = triton.language.zeros((BLOCK_M,), dtype=triton.language.float32)
    acc = triton.language.zeros((BLOCK_M, HEAD_DIM_), dtype=triton.language.float32)

    for start_n in range(0, num_tokens, BLOCK_N):
        k_rows = start_n + triton.language.arange(0, BLOCK_N)
        k_mask = k_rows < num_tokens
        k = triton.language.load(
            k_ptr + k_rows[:, None] * stride_km + kv_head * stride_kh + d[None, :],
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(triton.language.float32)
        v = triton.language.load(
            v_ptr + k_rows[:, None] * stride_vm + kv_head * stride_vh + d[None, :],
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(triton.language.float32)

        qk = triton.language.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
        causal = k_rows[None, :] <= q_rows[:, None]
        valid = q_mask[:, None] & k_mask[None, :] & causal
        qk = triton.language.where(valid, qk, -float("inf"))
        m_ij = triton.language.maximum(m_i, triton.language.max(qk, axis=1))
        alpha = triton.language.exp2((m_i - m_ij) * LOG2E_)
        p = triton.language.exp2((qk - m_ij[:, None]) * LOG2E_)
        l_i = l_i * alpha + triton.language.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += triton.language.sum(p[:, :, None] * v[None, :, :], axis=1)
        m_i = m_ij

    out = acc / l_i[:, None]
    triton.language.store(
        o_ptr + q_rows[:, None] * stride_om + q_head * stride_oh + d[None, :],
        out.to(triton.language.bfloat16),
        mask=q_mask[:, None] & d_mask[None, :],
    )


def run_candidate(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run the benchmark-only candidate into a caller-owned output buffer."""

    num_tokens = query.shape[0]
    grid = (triton.cdiv(num_tokens, 4), H_Q)
    _raw_causal_gqa_kernel[grid](
        query,
        key,
        value,
        output,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        output.stride(0),
        output.stride(1),
        num_tokens,
        1.0 / math.sqrt(HEAD_DIM),
        GROUP_SIZE_=GROUP_SIZE,
        LOG2E_=LOG2E,
        BLOCK_M=4,
        BLOCK_N=64,
        HEAD_DIM_=HEAD_DIM,
        num_warps=4,
    )
    return output


def run_math_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run the current explicit Math SDPA fallback into a fixed destination."""

    with sdpa_kernel(SDPBackend.MATH):
        result = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            is_causal=True,
            scale=1.0 / math.sqrt(HEAD_DIM),
            enable_gqa=True,
        )
    output.copy_(result[0].transpose(0, 1))
    return output


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - reference.float()
    return {
        "max_abs": float(delta.abs().max().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def fp64_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rows: list[int],
    heads: list[int],
) -> torch.Tensor:
    """Compute a small FP64 causal oracle without forming a full score matrix."""

    results = []
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for row in rows:
        row_results = []
        for head in heads:
            kv_head = head // GROUP_SIZE
            q = query[row, head].double()
            k = key[: row + 1, kv_head].double()
            v = value[: row + 1, kv_head].double()
            weights = torch.softmax(torch.matmul(k, q) * scale, dim=0)
            row_results.append(torch.matmul(weights, v))
        results.append(torch.stack(row_results))
    return torch.stack(results)


@dataclass(frozen=True)
class TimingConfig:
    warmups: int
    samples: int
    flush_mib: int


def timed(
    fn,
    flush: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    flush.zero_()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    output = fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1000.0, output


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def run_case(
    num_tokens: int,
    device: torch.device,
    config: TimingConfig,
    seed: int,
    case_index: int,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(seed)
    query = torch.randn(
        (num_tokens, H_Q, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    key = torch.randn(
        (num_tokens, H_K, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    value = torch.randn_like(key, generator=generator)
    candidate_output = torch.empty_like(query)
    math_output = torch.empty_like(query)
    flush = torch.empty(
        config.flush_mib * 1024 * 1024 // 4, dtype=torch.float32, device=device
    )

    # Compilation and warmup are outside the samples, but both paths receive
    # the same number of warmups before the first rotating sample.
    for _ in range(config.warmups):
        run_candidate(query, key, value, candidate_output)
        run_math_sdpa(query, key, value, math_output)
    torch.accelerator.synchronize()

    samples: dict[str, list[float]] = {"candidate": [], "current_math_sdpa": []}
    order: list[list[str]] = []
    runners = {
        "candidate": lambda: run_candidate(query, key, value, candidate_output),
        "current_math_sdpa": lambda: run_math_sdpa(query, key, value, math_output),
    }
    for sample_index in range(config.samples):
        names = ["candidate", "current_math_sdpa"]
        if (sample_index + case_index) % 2:
            names.reverse()
        order.append(names)
        for name in names:
            elapsed_us, _ = timed(runners[name], flush)
            samples[name].append(elapsed_us)

    # Correctness is checked after timing with the same output buffers.  The
    # oracle uses only first/middle/last rows and two heads, never MxM scores.
    rows = sorted({0, num_tokens // 2, num_tokens - 1})
    heads = [0, H_Q - 1]
    oracle = fp64_rows(query, key, value, rows, heads)
    candidate_selected = candidate_output[rows][:, heads]
    math_selected = math_output[rows][:, heads]
    candidate_error = error_metrics(candidate_selected, oracle)
    math_error = error_metrics(math_selected, oracle)
    candidate_error["finite"] = bool(torch.isfinite(candidate_output).all().item())
    candidate_speed = median(samples["current_math_sdpa"]) / median(
        samples["candidate"]
    )
    numeric_ratio_max = candidate_error["max_abs"] / max(math_error["max_abs"], 1e-12)
    numeric_ratio_rmse = candidate_error["rmse"] / max(math_error["rmse"], 1e-12)
    return {
        "q_len": num_tokens,
        "shape": {"Hq": H_Q, "Hk": H_K, "D": HEAD_DIM},
        "dtype": "bfloat16",
        "causal": True,
        "samples_us": samples,
        "median_us": {name: median(values) for name, values in samples.items()},
        "order": order,
        "oracle_rows": rows,
        "oracle_heads": heads,
        "math_error": math_error,
        "candidate_error": candidate_error,
        "candidate_to_math_error_ratio": {
            "max_abs": numeric_ratio_max,
            "rmse": numeric_ratio_rmse,
        },
        "speedup_math_over_candidate": candidate_speed,
        "numeric_gate": bool(
            candidate_error["finite"]
            and numeric_ratio_max <= 1.1
            and numeric_ratio_rmse <= 1.1
        ),
        "performance_gate": bool(candidate_speed >= 1.3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument(
        "--q-lens",
        type=int,
        nargs="+",
        default=[128, 129, 256, 512, 1024, 2048, 4096, 8192],
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    config = TimingConfig(args.warmups, args.samples, args.flush_mib)
    rows = []
    for case_index, q_len in enumerate(args.q_lens):
        print(f"running q_len={q_len}", flush=True)
        rows.append(
            run_case(
                q_len,
                device,
                config,
                seed=0x504341 + q_len,
                case_index=case_index,
            )
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    production_shapes = [row for row in rows if row["q_len"] in (256, 512)]
    speed_pass = bool(
        production_shapes
        and all(bool(row["performance_gate"]) for row in production_shapes)
    )
    numeric_pass = bool(rows and all(bool(row["numeric_gate"]) for row in rows))
    result = {
        "schema_version": 1,
        "contract": {
            "Hq": H_Q,
            "Hk": H_K,
            "D": HEAD_DIM,
            "gqa_group": GROUP_SIZE,
            "dtype": "bfloat16",
            "causal": True,
            "sinks": False,
            "sliding_window": False,
            "single_request": True,
        },
        "measurement": {
            "warmups": config.warmups,
            "samples": config.samples,
            "flush_mib": config.flush_mib,
            "output_preallocated": True,
            "rotating_order": True,
            "math_backend": "SDPBackend.MATH",
            "candidate": "online-softmax Triton benchmark-only",
        },
        "cases": rows,
        "gates": {
            "numeric_pass": numeric_pass,
            "production_shape_speed_pass": speed_pass,
            "status": "pass" if numeric_pass and speed_pass else "stop",
            "cold_32k_required": bool(numeric_pass and speed_pass),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(result["gates"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
