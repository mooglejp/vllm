# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate the installed AMD Triton FlashAttention backend for R5.

This is a benchmark-only harness.  It does not register an attention backend,
change TurboQuant dispatch, or modify the production K8/V4 cache contract.  A
case first compares the public FlashAttention varlen entry point with explicit
Math SDPA and an independent FP64 oracle.  Full-continuation timing is only
run when every requested numerical case passes the fixed R5 gate.

The effective input contract is the current large-continuation contract:
K8/V4 prefix -> existing dequant reader -> FP16 -> BF16, and raw BF16 current
K/V.  ``causal=True`` is used only after the bottom-right rectangular mask is
checked by the fixed ``cached_len=3, q_len=2`` fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# The host verification venv has the ROCm Triton driver, but its source
# worktree is not installed as a distribution.  vLLM consequently exposes a
# placeholder through ``vllm.triton_utils``.  Repair only those two imported
# symbols in this standalone process; production imports and dispatch are not
# changed.
from torch.nn.attention import SDPBackend, sdpa_kernel

import vllm.triton_utils as _vllm_triton_utils

triton = importlib.import_module("triton")
tl = importlib.import_module("triton.language")

if not hasattr(_vllm_triton_utils.tl, "float16"):
    _vllm_triton_utils.triton = triton
    _vllm_triton_utils.tl = tl


def _load_kv_helpers() -> tuple[Any, Any, Any, Any]:
    from benchmark_gfx1201_kv_format_diagnostic import (
        allocate_block_table,
        allocate_k8v4_cache,
        dequantize_k8v4_prefix,
        store_k8v4_cache,
    )

    return (
        allocate_block_table,
        allocate_k8v4_cache,
        dequantize_k8v4_prefix,
        store_k8v4_cache,
    )


(
    allocate_block_table,
    allocate_k8v4_cache,
    dequantize_k8v4_prefix,
    store_k8v4_cache,
) = _load_kv_helpers()

H_Q = 24
H_K = 4
HEAD_DIM = 256
GQA = H_Q // H_K
BLOCK_SIZE = 16
SCALE = HEAD_DIM**-0.5
UPSTREAM_ATOL = 1e-2
UPSTREAM_RTOL = 1e-2
NUMERIC_MARGIN = 1.10


@dataclass(frozen=True)
class Case:
    cached_len: int
    q_len: int

    @property
    def seq_len(self) -> int:
        return self.cached_len + self.q_len


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backend_provenance() -> dict[str, Any]:
    """Describe the package and entry point actually selected at import time."""
    try:
        dist = importlib.metadata.distribution("flash-attn")
        package_version = dist.version
        direct_url = dist.read_text("direct_url.json")
        direct_url_data = json.loads(direct_url) if direct_url else None
    except (importlib.metadata.PackageNotFoundError, OSError, json.JSONDecodeError):
        package_version = None
        direct_url_data = None

    import flash_attn
    import flash_attn.flash_attn_interface as interface

    entry = flash_attn.flash_attn_varlen_func
    amd_interface = (
        Path(interface.__file__).parent / "flash_attn_triton_amd" / "interface_fa.py"
    )
    prefill = amd_interface.parent / "fwd_prefill.py"
    test_file = amd_interface.parent / "test.py"
    return {
        "package": "flash_attn",
        "version": package_version,
        "package_file": str(Path(flash_attn.__file__).resolve()),
        "interface_file": str(Path(interface.__file__).resolve()),
        "entry_point": "flash_attn.flash_attn_varlen_func",
        "entry_source": inspect.getsourcefile(entry),
        "amd_selector": os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE"),
        "amd_use_triton_rocm": bool(getattr(interface, "USE_TRITON_ROCM", False)),
        "amd_interface_file": str(amd_interface),
        "amd_interface_sha256": _sha256(amd_interface)
        if amd_interface.exists()
        else None,
        "amd_prefill_file": str(prefill),
        "amd_prefill_sha256": _sha256(prefill) if prefill.exists() else None,
        "upstream_test_file": str(test_file),
        "upstream_test_sha256": _sha256(test_file) if test_file.exists() else None,
        "distribution_direct_url": direct_url_data,
        "source_commit": None,
        "source_commit_note": "wheel metadata did not expose an upstream commit",
    }


def sync() -> None:
    torch.accelerator.synchronize()


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    delta = actual.float() - reference.float()
    ref_float = reference.float()
    ref_norm = ref_float.norm().item()
    return {
        "max_abs": float(delta.abs().max().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "relative_l2": float(delta.norm().item() / max(ref_norm, 1e-12)),
        "finite": bool(torch.isfinite(actual).all().item()),
        "mismatch_elements": int(torch.count_nonzero(actual != reference).item()),
        "elements": int(actual.numel()),
    }


def _make_inputs(
    case: Case, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    query = torch.randn(
        case.q_len,
        H_Q,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    key = torch.randn(
        case.seq_len,
        H_K,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    value = torch.randn(
        case.seq_len,
        H_K,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    return query, key, value, key[: case.cached_len], value[: case.cached_len]


def _materialize_contract(
    case: Case,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build BF16 K/V under the existing K8/V4-prefix/raw-current contract."""
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

    dtype = torch.bfloat16
    k_full = torch.empty(case.seq_len, H_K, HEAD_DIM, dtype=dtype, device=key.device)
    v_full = torch.empty_like(k_full)
    if case.cached_len == 0:
        k_full.copy_(key)
        v_full.copy_(value)
        return (
            k_full,
            v_full,
            {
                "prefix": "none",
                "current": "raw BF16",
                "prefix_fp16_roundtrip": False,
            },
        )

    config = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", HEAD_DIM)
    block_table, _ = allocate_block_table(case.seq_len, key.device)
    cache = allocate_k8v4_cache(case.seq_len, config, key.device)
    pit = torch.eye(HEAD_DIM, dtype=torch.float32, device=key.device)
    midpoints = torch.zeros(
        config.n_centroids - 1, dtype=torch.float32, device=key.device
    )
    centroids = torch.ones(config.n_centroids, dtype=torch.float32, device=key.device)
    store_k8v4_cache(
        key,
        value,
        cache,
        block_table,
        config,
        pit,
        midpoints,
        centroids,
    )
    alloc_len = math.ceil(case.cached_len / BLOCK_SIZE) * BLOCK_SIZE
    decoded_k = torch.empty(
        1, H_K, alloc_len, HEAD_DIM, dtype=torch.float16, device=key.device
    )
    decoded_v = torch.empty_like(decoded_k)
    dequantize_k8v4_prefix(
        cache,
        block_table,
        case.cached_len,
        config,
        centroids,
        decoded_k,
        decoded_v,
    )
    # The production contract writes the reader result in FP16 and converts it
    # to query dtype only while assembling the dense attention inputs.
    k_full[: case.cached_len].copy_(
        decoded_k[0, :, : case.cached_len].transpose(0, 1).to(dtype)
    )
    v_full[: case.cached_len].copy_(
        decoded_v[0, :, : case.cached_len].transpose(0, 1).to(dtype)
    )
    k_full[case.cached_len :].copy_(key[case.cached_len :])
    v_full[case.cached_len :].copy_(value[case.cached_len :])
    sync()
    return (
        k_full,
        v_full,
        {
            "prefix": "K8/V4 SoA cache -> existing reader FP16 -> BF16",
            "current": "raw BF16",
            "prefix_fp16_roundtrip": True,
        },
    )


def causal_mask(case: Case, device: torch.device) -> torch.Tensor:
    q_pos = torch.arange(case.q_len, device=device).unsqueeze(1) + case.cached_len
    k_pos = torch.arange(case.seq_len, device=device).unsqueeze(0)
    return k_pos <= q_pos


def run_math(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
            scale=SCALE,
            enable_gqa=True,
        )[0].transpose(0, 1)


def run_flash(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    from flash_attn import flash_attn_varlen_func

    q_len = query.shape[0]
    seq_len = key.shape[0]
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=query.device)
    cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=query.device)
    return flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=q_len,
        max_seqlen_k=seq_len,
        dropout_p=0.0,
        softmax_scale=SCALE,
        causal=True,
    )


def _oracle_rows(case: Case) -> tuple[list[int], list[int], bool]:
    full = case.cached_len <= 3 and case.q_len <= 129
    rows = (
        list(range(case.q_len))
        if full
        else sorted({0, case.q_len // 2, case.q_len - 1})
    )
    heads = list(range(H_Q)) if full else [0, H_Q // 2, H_Q - 1]
    return rows, heads, full


def fp64_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    case: Case,
) -> dict[tuple[int, int], torch.Tensor]:
    rows, heads, _ = _oracle_rows(case)
    # ROCm gfx1201 does not provide a usable FP64 GEMV image in every
    # validation environment.  Copy only the selected reference inputs to CPU
    # so the oracle remains independent of the candidate GPU reduction.
    query64 = query.detach().cpu().double()
    key64 = key.detach().cpu().double()
    value64 = value.detach().cpu().double()
    oracle: dict[tuple[int, int], torch.Tensor] = {}
    for row in rows:
        visible = case.cached_len + row + 1
        for head in heads:
            kv_head = head // GQA
            scores = torch.matmul(key64[:visible, kv_head], query64[row, head]) * SCALE
            weights = torch.softmax(scores, dim=0)
            oracle[(row, head)] = torch.matmul(weights, value64[:visible, kv_head])
    return oracle


def _select(
    output: torch.Tensor, oracle: dict[tuple[int, int], torch.Tensor]
) -> torch.Tensor:
    return torch.stack(
        [output[row, head].detach().cpu().double() for row, head in oracle]
    )


def oracle_metrics(
    output: torch.Tensor,
    oracle: dict[tuple[int, int], torch.Tensor],
) -> dict[str, Any]:
    actual = _select(output, oracle)
    reference64 = torch.stack(list(oracle.values()))
    reference_bf16 = reference64.to(torch.bfloat16)
    pre_cast = error_metrics(actual, reference64)
    final_cast = error_metrics(actual.to(torch.bfloat16), reference_bf16)
    return {
        "precast_fp64": pre_cast,
        "final_bf16": final_cast,
        "oracle_row_count": len(oracle),
    }


def upstream_close(
    output: torch.Tensor,
    oracle: dict[tuple[int, int], torch.Tensor],
) -> dict[str, Any]:
    actual = _select(output, oracle).to(torch.bfloat16)
    reference = torch.stack(list(oracle.values())).to(torch.bfloat16)
    try:
        torch.testing.assert_close(
            actual,
            reference,
            atol=UPSTREAM_ATOL,
            rtol=UPSTREAM_RTOL,
            equal_nan=False,
        )
    except AssertionError as exc:
        return {
            "pass": False,
            "atol": UPSTREAM_ATOL,
            "rtol": UPSTREAM_RTOL,
            "error": str(exc).splitlines()[0],
        }
    return {"pass": True, "atol": UPSTREAM_ATOL, "rtol": UPSTREAM_RTOL}


def numeric_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_bf16 = candidate["final_bf16"]
    baseline_bf16 = baseline["final_bf16"]
    max_limit = float(baseline_bf16["max_abs"]) * NUMERIC_MARGIN
    rmse_limit = float(baseline_bf16["rmse"]) * NUMERIC_MARGIN
    max_pass = float(candidate_bf16["max_abs"]) <= max_limit
    rmse_pass = float(candidate_bf16["rmse"]) <= rmse_limit
    return {
        "margin": NUMERIC_MARGIN,
        "max_abs_limit": max_limit,
        "rmse_limit": rmse_limit,
        "max_abs_pass": max_pass,
        "rmse_pass": rmse_pass,
        "pass": bool(candidate_bf16["finite"] and max_pass and rmse_pass),
    }


def boundary_fixture(device: torch.device) -> dict[str, Any]:
    """Check the documented q=2/k=5 bottom-right mask directly."""
    case = Case(cached_len=3, q_len=2)
    query = torch.zeros(case.q_len, H_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    key = torch.zeros(case.seq_len, H_K, HEAD_DIM, dtype=torch.bfloat16, device=device)
    value = torch.zeros_like(key)
    value[:, :, 0] = torch.arange(case.seq_len, dtype=torch.bfloat16, device=device)[
        :, None
    ]
    expected = torch.tensor([1.5, 2.0], dtype=torch.bfloat16, device=device)
    mask = causal_mask(case, device)
    math_out = run_math(query, key, value, mask)
    flash_out = run_flash(query, key, value)
    sync()
    math_ok = bool(torch.equal(math_out[:, 0, 0], expected))
    flash_ok = bool(torch.equal(flash_out[:, 0, 0], expected))
    return {
        "case": {"cached_len": 3, "q_len": 2, "seq_len": 5},
        "mask": [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
        "expected_head0_dim0": [1.5, 2.0],
        "math_head0_dim0": [float(x) for x in math_out[:, 0, 0].cpu()],
        "flash_head0_dim0": [float(x) for x in flash_out[:, 0, 0].cpu()],
        "math_pass": math_ok,
        "flash_pass": flash_ok,
        "pass": math_ok and flash_ok,
    }


def _time_samples(
    run: Callable[[], torch.Tensor],
    warmups: int,
    samples: int,
    flush: torch.Tensor,
) -> list[float]:
    for _ in range(warmups):
        run()
    sync()
    values: list[float] = []
    for _ in range(samples):
        flush.zero_()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        sync()
        values.append(start.elapsed_time(end) * 1000.0)
    return values


def benchmark_case(
    case: Case,
    device: torch.device,
    seed: int,
    warmups: int,
    samples: int,
    flush: torch.Tensor,
    measure: bool = False,
) -> dict[str, Any]:
    query, key, value, _, _ = _make_inputs(case, device, seed)
    k_full, v_full, materialization = _materialize_contract(case, key, value)
    mask = causal_mask(case, device)
    oracle = fp64_oracle(query, k_full, v_full, case)
    baseline = run_math(query, k_full, v_full, mask)
    candidate = run_flash(query, k_full, v_full)
    sync()
    baseline_metrics = oracle_metrics(baseline, oracle)
    candidate_metrics = oracle_metrics(candidate, oracle)
    baseline_close = upstream_close(baseline, oracle)
    candidate_close = upstream_close(candidate, oracle)
    gate = numeric_gate(candidate_metrics, baseline_metrics)
    row: dict[str, Any] = {
        "case": {
            "cached_len": case.cached_len,
            "q_len": case.q_len,
            "seq_len": case.seq_len,
        },
        "seed": seed,
        "shape": {"Hq": H_Q, "Hk": H_K, "D": HEAD_DIM, "GQA": GQA},
        "dtype": "torch.bfloat16",
        "scale": SCALE,
        "causal": True,
        "materialization": materialization,
        "oracle_scope": {
            "full_output": _oracle_rows(case)[2],
            "row_count": len(oracle),
            "rows": (
                [list(row) for row in oracle] if not _oracle_rows(case)[2] else None
            ),
        },
        "baseline_math": baseline_metrics,
        "candidate_flash_triton_amd": candidate_metrics,
        "official_upstream_forward_criterion": {
            "baseline": baseline_close,
            "candidate": candidate_close,
            "source": "flash_attn_triton_amd/test.py",
            "formula": "torch.testing.assert_close",
        },
        "numeric_gate": gate,
        "status": "numeric_pass"
        if gate["pass"] and candidate_close["pass"]
        else "numeric_fail",
    }
    if measure and gate["pass"] and candidate_close["pass"]:

        def attention_math() -> torch.Tensor:
            return run_math(query, k_full, v_full, mask)

        def attention_flash() -> torch.Tensor:
            return run_flash(query, k_full, v_full)

        def full_math() -> torch.Tensor:
            materialized_k, materialized_v, _ = _materialize_contract(case, key, value)
            return run_math(query, materialized_k, materialized_v, mask)

        def full_flash() -> torch.Tensor:
            materialized_k, materialized_v, _ = _materialize_contract(case, key, value)
            return run_flash(query, materialized_k, materialized_v)

        methods = {
            "attention_math": attention_math,
            "attention_flash_triton_amd": attention_flash,
            "continuation_math": full_math,
            "continuation_flash_triton_amd": full_flash,
        }
        timings: dict[str, Any] = {}
        names = list(methods)
        for index in range(warmups + samples):
            order = names if (index + seed) % 2 == 0 else list(reversed(names))
            if index < warmups:
                for name in order:
                    methods[name]()
                continue
            for name in order:
                timings.setdefault(name, []).append(
                    _time_samples(methods[name], 0, 1, flush)[0]
                )
        row["timing"] = {
            name: {
                "samples_us": values,
                "median_us": statistics.median(values),
                "min_us": min(values),
                "p95_us": sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)],
            }
            for name, values in timings.items()
        }
    else:
        row["timing"] = {
            "status": "not_measured_until_all_numerical_cases_pass"
            if not measure
            else "skipped_numeric_gate"
        }
    return row


def run_trace(device: torch.device, case: Case, seed: int) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    query, key, value, _, _ = _make_inputs(case, device, seed)
    k_full, v_full, _ = _materialize_contract(case, key, value)
    run_flash(query, k_full, v_full)
    sync()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        run_flash(query, k_full, v_full)
        sync()
    events = []
    for event in prof.events():
        if event.device_type is not ProfilerActivity.CPU:
            events.append(
                {
                    "name": event.name,
                    "device_time_us": float(event.device_time_total),
                }
            )
    return {
        "case": {"cached_len": case.cached_len, "q_len": case.q_len},
        "events": events,
        "candidate_kernel_names": sorted(
            {item["name"] for item in events if "attn" in item["name"].lower()}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cached-lens", type=int, nargs="+", default=[0, 3, 4096])
    parser.add_argument(
        "--q-lens", type=int, nargs="+", default=[1, 127, 128, 129, 256, 512]
    )
    parser.add_argument("--long-cached-len", type=int, default=32768)
    parser.add_argument("--include-long", action="store_true")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("R5 requires a ROCm GPU")
    if os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE") != "TRUE":
        raise RuntimeError("FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE is required")
    device = torch.device("cuda")
    provenance = backend_provenance()
    if not provenance["amd_use_triton_rocm"]:
        raise RuntimeError("flash_attn did not select its AMD Triton implementation")

    cases = [
        Case(cached, q_len) for cached in args.cached_lens for q_len in args.q_lens
    ]
    if args.include_long:
        cases.append(Case(args.long_cached_len, 256))
    flush = torch.empty(args.flush_mib * 2**20, dtype=torch.uint8, device=device)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "start_commit": "c8b800ef786230644ac29bb0a3bd9b1b9c904a5f",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "properties": str(torch.cuda.get_device_properties(device)),
        },
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "backend": provenance,
        "contract": {
            "dtype": "BF16",
            "Hq": H_Q,
            "Hk": H_K,
            "GQA": GQA,
            "D": HEAD_DIM,
            "dropout": 0.0,
            "causal": "bottom-right rectangular: j <= cached_len + i",
            "prefix": "same synthetic K8/V4 SoA cache, existing reader FP16 -> BF16",
            "current": "raw BF16 K/V",
            "prefix_requantization": False,
            "sinks": False,
            "sliding_window": False,
        },
        "numerical_manifest": {
            "reference": "independent FP64 online row oracle from effective BF16 K/V",
            "final_reference": "FP64 oracle cast to BF16",
            "official_upstream_atol": UPSTREAM_ATOL,
            "official_upstream_rtol": UPSTREAM_RTOL,
            "official_source": "flash_attn_triton_amd/test.py in loaded wheel",
            "candidate_vs_current_margin": NUMERIC_MARGIN,
            "gate": "candidate final BF16 max-abs and RMSE <= 1.10x explicit Math SDPA",
        },
        "measurement_manifest": {
            "attention_only": "prepared effective BF16 K/V -> output",
            "continuation": (
                "K8/V4 dequant + FP16->BF16 assembly + raw current K/V + attention"
            ),
            "warmups": args.warmups,
            "samples": args.samples,
            "rotating_order": True,
            "flush_mib": args.flush_mib,
            "timing_status": "conditional on all numerical gates",
        },
        "boundary_fixture": boundary_fixture(device),
        "cases": [],
    }
    for index, case in enumerate(cases):
        print(
            f"[{index + 1}/{len(cases)}] cached={case.cached_len} q={case.q_len}",
            flush=True,
        )
        try:
            result = benchmark_case(
                case,
                device,
                args.seed + index,
                args.warmups,
                args.samples,
                flush,
                measure=False,
            )
        except torch.OutOfMemoryError as exc:
            result = {
                "case": {"cached_len": case.cached_len, "q_len": case.q_len},
                "status": "oom",
                "error": str(exc),
            }
            torch.accelerator.empty_cache()
        except Exception as exc:
            result = {
                "case": {"cached_len": case.cached_len, "q_len": case.q_len},
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            torch.accelerator.empty_cache()
        report["cases"].append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    complete = [
        item for item in report["cases"] if item.get("status") == "numeric_pass"
    ]
    numeric_pass = bool(complete) and len(complete) == len(report["cases"])
    report["numeric_gate_summary"] = {
        "all_cases_complete": len(complete) == len(report["cases"]),
        "all_cases_pass": numeric_pass,
        "case_count": len(report["cases"]),
    }
    if numeric_pass:
        for index, case in enumerate(cases):
            timed = benchmark_case(
                case,
                device,
                args.seed + index,
                args.warmups,
                args.samples,
                flush,
                measure=True,
            )
            report["cases"][index]["timing"] = timed["timing"]
    if args.trace:
        report["trace"] = run_trace(device, Case(4096, 128), args.seed + 9000)
    report["status"] = (
        "numeric_pass_speed_pending" if numeric_pass else "numeric_gate_failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
