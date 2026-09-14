# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""R5 speed evaluation after the strict Math-equivalence diagnostic.

The strict Math ratio from ``benchmark_gfx1201_r5_attention.py`` remains a
diagnostic artifact.  This entry point uses the loaded AMD Triton upstream
tolerance, full-output finiteness, the documented bottom-right causal fixture,
and the K8/V4-prefix/raw-current input contract as prerequisites for timing.
It never changes a production dispatch or writes a new attention kernel.

Cache allocation and store are performed once during preparation.  Attention
timing uses prepared dense BF16 K/V and fixed metadata.  Continuation timing
reuses the prepared cache but includes each invocation's K8/V4 decode,
FP16-to-BF16 conversion, raw-current assembly, mask/metadata allocation,
attention, and output allocation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import benchmark_gfx1201_r5_attention as strict
import torch

UPSTREAM_ATOL = strict.UPSTREAM_ATOL
UPSTREAM_RTOL = strict.UPSTREAM_RTOL
SPEED_GATE = 1.30
DEFAULT_CASES = ((4096, 256), (32768, 256))


@dataclass
class PreparedCase:
    case: strict.Case
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    config: Any
    block_table: torch.Tensor
    cache: Any
    centroids: torch.Tensor
    k_full: torch.Tensor
    v_full: torch.Tensor
    mask: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    alloc_len: int


def _fp64_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Aggregate selected oracle rows without converting the difference to FP32."""
    actual64 = actual.detach().cpu().double()
    reference64 = reference.detach().cpu().double()
    delta = actual64 - reference64
    ref_norm = reference64.norm().item()
    return {
        "max_abs": float(delta.abs().max().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "relative_l2": float(delta.norm().item() / max(ref_norm, 1e-12)),
        "finite": bool(torch.isfinite(actual64).all().item()),
        "mismatch_elements": int(torch.count_nonzero(delta != 0).item()),
        "elements": int(actual64.numel()),
    }


def _selected_fp64_metrics(
    output: torch.Tensor,
    oracle: dict[tuple[int, int], torch.Tensor],
) -> dict[str, Any]:
    actual = strict._select(output, oracle)
    reference = torch.stack(list(oracle.values()))
    rounded_actual = actual.to(torch.bfloat16).to(torch.float64)
    rounded_reference = reference.to(torch.bfloat16).to(torch.float64)
    return {
        "fp64_reference_difference": _fp64_error(actual, reference),
        "final_bf16_reference_difference": _fp64_error(
            rounded_actual, rounded_reference
        ),
        "oracle_row_count": len(oracle),
        "note": (
            "fp64_reference_difference is an independent reference delta; it is "
            "not a kernel-internal pre-cast output."
        ),
    }


def _upstream_tolerance_check(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, Any]:
    """Apply the loaded AMD test tolerance to the effective BF16 contract."""
    try:
        torch.testing.assert_close(
            candidate,
            reference,
            atol=UPSTREAM_ATOL,
            rtol=UPSTREAM_RTOL,
            equal_nan=True,
        )
    except AssertionError as exc:
        return {
            "pass": False,
            "atol": UPSTREAM_ATOL,
            "rtol": UPSTREAM_RTOL,
            "error": str(exc).splitlines()[0],
        }
    return {"pass": True, "atol": UPSTREAM_ATOL, "rtol": UPSTREAM_RTOL}


def _strict_math_diagnostic(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    candidate_m = candidate["final_bf16_reference_difference"]
    baseline_m = baseline["final_bf16_reference_difference"]

    def ratio(candidate_value: float, baseline_value: float) -> float | None:
        if baseline_value == 0.0:
            return None if candidate_value == 0.0 else float("inf")
        return candidate_value / baseline_value

    max_ratio = ratio(candidate_m["max_abs"], baseline_m["max_abs"])
    rmse_ratio = ratio(candidate_m["rmse"], baseline_m["rmse"])
    return {
        "max_abs_ratio": max_ratio,
        "rmse_ratio": rmse_ratio,
        "margin": 1.10,
        "pass": bool(
            max_ratio is not None
            and rmse_ratio is not None
            and max_ratio <= 1.10
            and rmse_ratio <= 1.10
        ),
        "note": "Diagnostic only; this failure does not block speed measurement.",
    }


def _materialize_from_existing(
    prepared: PreparedCase,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode retained cache state and assemble the current raw chunk."""
    device = prepared.query.device
    decoded_k = torch.empty(
        1,
        strict.H_K,
        prepared.alloc_len,
        strict.HEAD_DIM,
        dtype=torch.float16,
        device=device,
    )
    decoded_v = torch.empty_like(decoded_k)
    strict.dequantize_k8v4_prefix(
        prepared.cache,
        prepared.block_table,
        prepared.case.cached_len,
        prepared.config,
        prepared.centroids,
        decoded_k,
        decoded_v,
    )
    k_full = torch.empty(
        prepared.case.seq_len,
        strict.H_K,
        strict.HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    v_full = torch.empty_like(k_full)
    k_full[: prepared.case.cached_len].copy_(
        decoded_k[0, :, : prepared.case.cached_len].transpose(0, 1).to(torch.bfloat16)
    )
    v_full[: prepared.case.cached_len].copy_(
        decoded_v[0, :, : prepared.case.cached_len].transpose(0, 1).to(torch.bfloat16)
    )
    k_full[prepared.case.cached_len :].copy_(prepared.key[prepared.case.cached_len :])
    v_full[prepared.case.cached_len :].copy_(prepared.value[prepared.case.cached_len :])
    return k_full, v_full


def _prepare_case(case: strict.Case, device: torch.device, seed: int) -> PreparedCase:
    query, key, value, _, _ = strict._make_inputs(case, device, seed)
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

    config = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", strict.HEAD_DIM)
    block_table, _ = strict.allocate_block_table(case.seq_len, device)
    cache = strict.allocate_k8v4_cache(case.seq_len, config, device)
    pit = torch.eye(strict.HEAD_DIM, dtype=torch.float32, device=device)
    midpoints = torch.zeros(config.n_centroids - 1, dtype=torch.float32, device=device)
    centroids = torch.ones(config.n_centroids, dtype=torch.float32, device=device)
    strict.store_k8v4_cache(
        key,
        value,
        cache,
        block_table,
        config,
        pit,
        midpoints,
        centroids,
    )
    alloc_len = math.ceil(case.cached_len / strict.BLOCK_SIZE) * strict.BLOCK_SIZE
    prepared = PreparedCase(
        case=case,
        query=query,
        key=key,
        value=value,
        config=config,
        block_table=block_table,
        cache=cache,
        centroids=centroids,
        k_full=None,  # type: ignore[arg-type]
        v_full=None,  # type: ignore[arg-type]
        mask=strict.causal_mask(case, device),
        cu_q=torch.tensor([0, case.q_len], dtype=torch.int32, device=device),
        cu_k=torch.tensor([0, case.seq_len], dtype=torch.int32, device=device),
        alloc_len=alloc_len,
    )
    prepared.k_full, prepared.v_full = _materialize_from_existing(prepared)
    strict.sync()
    return prepared


def _flash_prepared(
    prepared: PreparedCase, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    return _flash_with_metadata(prepared, key, value, prepared.cu_q, prepared.cu_k)


def _flash_with_metadata(
    prepared: PreparedCase,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
) -> torch.Tensor:
    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(
        q=prepared.query,
        k=key,
        v=value,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=prepared.case.q_len,
        max_seqlen_k=prepared.case.seq_len,
        dropout_p=0.0,
        softmax_scale=strict.SCALE,
        causal=True,
    )


def _math_prepared(
    prepared: PreparedCase, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    return strict.run_math(prepared.query, key, value, prepared.mask)


def _numeric_case(prepared: PreparedCase) -> dict[str, Any]:
    baseline = _math_prepared(prepared, prepared.k_full, prepared.v_full)
    candidate = _flash_prepared(prepared, prepared.k_full, prepared.v_full)
    strict.sync()
    oracle = strict.fp64_oracle(
        prepared.query, prepared.k_full, prepared.v_full, prepared.case
    )
    baseline_metrics = _selected_fp64_metrics(baseline, oracle)
    candidate_metrics = _selected_fp64_metrics(candidate, oracle)
    baseline_finite = bool(torch.isfinite(baseline).all().item())
    candidate_finite = bool(torch.isfinite(candidate).all().item())
    oracle_finite = all(torch.isfinite(value).all().item() for value in oracle.values())
    upstream = _upstream_tolerance_check(
        candidate.to(torch.bfloat16), baseline.to(torch.bfloat16)
    )
    contract = {
        "query": {
            "shape": list(prepared.query.shape),
            "dtype": str(prepared.query.dtype),
            "stride": list(prepared.query.stride()),
        },
        "key": {
            "shape": list(prepared.key.shape),
            "dtype": str(prepared.key.dtype),
            "stride": list(prepared.key.stride()),
        },
        "value": {
            "shape": list(prepared.value.shape),
            "dtype": str(prepared.value.dtype),
            "stride": list(prepared.value.stride()),
        },
        "prefix": "retained K8/V4 SoA cache -> FP16 -> BF16",
        "cache": {
            "dtype": str(prepared.cache.dtype),
            "shape": list(prepared.cache.shape),
            "block_table_dtype": str(prepared.block_table.dtype),
            "block_table_shape": list(prepared.block_table.shape),
        },
        "current": "raw BF16 K/V",
        "scale": strict.SCALE,
        "causal_rule": "j <= cached_len + query_row",
        "prefix_requantization": False,
    }
    input_contract_pass = bool(
        prepared.query.dtype == torch.bfloat16
        and prepared.key.dtype == torch.bfloat16
        and prepared.value.dtype == torch.bfloat16
        and tuple(prepared.query.shape)
        == (prepared.case.q_len, strict.H_Q, strict.HEAD_DIM)
        and tuple(prepared.key.shape)
        == (prepared.case.seq_len, strict.H_K, strict.HEAD_DIM)
        and tuple(prepared.value.shape)
        == (prepared.case.seq_len, strict.H_K, strict.HEAD_DIM)
        and prepared.query.stride(-1) == 1
        and prepared.key.stride(-1) == 1
        and prepared.value.stride(-1) == 1
    )
    numeric_pass = bool(
        input_contract_pass
        and baseline_finite
        and candidate_finite
        and oracle_finite
        and upstream["pass"]
    )
    return {
        "case": {
            "cached_len": prepared.case.cached_len,
            "q_len": prepared.case.q_len,
            "seq_len": prepared.case.seq_len,
        },
        "contract": contract,
        "input_contract_pass": input_contract_pass,
        "finite": {
            "baseline_math_full_output": baseline_finite,
            "candidate_flash_triton_amd_full_output": candidate_finite,
            "selected_fp64_oracle": oracle_finite,
        },
        "baseline_math": baseline_metrics,
        "candidate_flash_triton_amd": candidate_metrics,
        "upstream_tolerance_check": {
            "reference": "explicit Math SDPA over the same effective BF16 K/V",
            "source_dtype": "torch.float16 in loaded upstream test",
            "evaluation_dtype": "torch.bfloat16 (R5 contract adaptation)",
            "source_test": "flash_attn_triton_amd/test.py",
            "formula": "torch.testing.assert_close",
            "atol": UPSTREAM_ATOL,
            "rtol": UPSTREAM_RTOL,
            "equal_nan": True,
            "result": upstream,
        },
        "strict_math_ratio_diagnostic": _strict_math_diagnostic(
            candidate_metrics, baseline_metrics
        ),
        "status": "numeric_pass" if numeric_pass else "numeric_fail",
        "timing": {"status": "pending_all_numeric_cases"},
    }


def _timed_full(prepared: PreparedCase, method: str) -> torch.Tensor:
    key, value = _materialize_from_existing(prepared)
    # These metadata objects are intentionally inside the full-continuation
    # timing boundary.  The attention-only path uses prepared metadata below.
    if method == "math":
        mask = strict.causal_mask(prepared.case, prepared.query.device)
        return strict.run_math(prepared.query, key, value, mask)
    cu_q = torch.tensor(
        [0, prepared.case.q_len], dtype=torch.int32, device=prepared.query.device
    )
    cu_k = torch.tensor(
        [0, prepared.case.seq_len], dtype=torch.int32, device=prepared.query.device
    )
    return _flash_with_metadata(prepared, key, value, cu_q, cu_k)


def _time_samples(
    methods: dict[str, Callable[[], torch.Tensor]],
    warmups: int,
    samples: int,
    flush: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    names = list(methods)
    for index in range(warmups):
        order = names if (index + seed) % 2 == 0 else list(reversed(names))
        for name in order:
            methods[name]()
    strict.sync()
    values: dict[str, list[float]] = {name: [] for name in names}
    for index in range(samples):
        order = names if (index + seed) % 2 == 0 else list(reversed(names))
        for name in order:
            flush.zero_()
            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)
            start.record()
            methods[name]()
            end.record()
            strict.sync()
            values[name].append(start.elapsed_time(end) * 1000.0)
    return {
        name: {
            "samples_us": samples_for_name,
            "median_us": statistics.median(samples_for_name),
            "min_us": min(samples_for_name),
            "p95_us": sorted(samples_for_name)[
                max(0, math.ceil(0.95 * len(samples_for_name)) - 1)
            ],
        }
        for name, samples_for_name in values.items()
    }


def _timing_case(
    prepared: PreparedCase, warmups: int, samples: int, flush: torch.Tensor, seed: int
) -> dict[str, Any]:
    methods = {
        "attention_math": lambda: _math_prepared(
            prepared, prepared.k_full, prepared.v_full
        ),
        "attention_flash_triton_amd": lambda: _flash_prepared(
            prepared, prepared.k_full, prepared.v_full
        ),
        "continuation_math": lambda: _timed_full(prepared, "math"),
        "continuation_flash_triton_amd": lambda: _timed_full(prepared, "flash"),
    }
    return _time_samples(methods, warmups, samples, flush, seed)


def _speed_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for item in cases:
        timing = item.get("timing", {})
        if "continuation_math" not in timing:
            continue
        case = item["case"]
        key = f"cached{case['cached_len']}_q{case['q_len']}"
        baseline = timing["continuation_math"]["median_us"]
        candidate = timing["continuation_flash_triton_amd"]["median_us"]
        summary[key] = {
            "baseline_continuation_median_us": baseline,
            "candidate_continuation_median_us": candidate,
            "speedup": baseline / candidate,
            "gate": baseline / candidate >= SPEED_GATE,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    if not torch.accelerator.is_available():
        raise RuntimeError("R5 requires a ROCm GPU")
    if strict.os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE") != "TRUE":
        raise RuntimeError("FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE is required")
    device = torch.device("cuda")
    provenance = strict.backend_provenance()
    if not provenance["amd_use_triton_rocm"]:
        raise RuntimeError("flash_attn did not select its AMD Triton implementation")

    boundary = strict.boundary_fixture(device)
    flush = torch.empty(args.flush_mib * 2**20, dtype=torch.uint8, device=device)
    report: dict[str, Any] = {
        "schema_version": 2,
        "evaluation": "R5 relaxed upstream-tolerance speed evaluation",
        "strict_math_record": (
            "66cd1152ab remains the strict Math-ratio diagnostic; its failure is "
            "not overwritten and does not block this speed measurement."
        ),
        "start_commit": "66cd1152ab",
        "gpu": {
            "device": str(device),
            "accelerator": str(torch.accelerator.current_accelerator()),
        },
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "backend": provenance,
        "contract": {
            "dtype": "BF16",
            "Hq": strict.H_Q,
            "Hk": strict.H_K,
            "GQA": strict.GQA,
            "D": strict.HEAD_DIM,
            "dropout": 0.0,
            "causal": "bottom-right rectangular: j <= cached_len + i",
            "prefix": "retained K8/V4 SoA cache -> existing reader FP16 -> BF16",
            "current": "raw BF16 K/V",
            "scale": strict.SCALE,
            "production_changes": False,
        },
        "upstream_tolerance_manifest": {
            "source": "loaded flash_attn_triton_amd/test.py",
            "source_dtype": "torch.float16",
            "source_reference": "attention_forward_pytorch_ref_impl",
            "evaluation_dtype": "torch.bfloat16",
            "evaluation_reference": "explicit Math SDPA over effective BF16 K/V",
            "atol": UPSTREAM_ATOL,
            "rtol": UPSTREAM_RTOL,
            "equal_nan": True,
            "adaptation_note": (
                "The same upstream tolerances are applied to the R5 BF16 contract; "
                "this is recorded as an adaptation, not as the upstream FP16 test "
                "itself."
            ),
        },
        "measurement_manifest": {
            "cases": [
                {"cached_len": cached, "q_len": q} for cached, q in DEFAULT_CASES
            ],
            "warmups": args.warmups,
            "samples": args.samples,
            "rotating_order": True,
            "flush_mib": args.flush_mib,
            "attention_only": (
                "prepared effective BF16 K/V, fixed mask/cu_seqlens, attention output"
            ),
            "continuation": (
                "retained cache decode + FP16->BF16 + raw-current assembly + "
                "mask/metadata/output allocation + attention"
            ),
            "preparation_excluded": "cache allocation, cache store, fixed metadata",
            "speed_gate": "32K/q256 continuation speedup >= 1.30x",
        },
        "boundary_fixture": boundary,
        "cases": [],
    }

    prepared_cases: list[PreparedCase] = []
    for index, (cached_len, q_len) in enumerate(DEFAULT_CASES):
        case = strict.Case(cached_len=cached_len, q_len=q_len)
        print(f"numeric cached={cached_len} q={q_len}", flush=True)
        prepared = _prepare_case(case, device, args.seed + index)
        prepared_cases.append(prepared)
        result = _numeric_case(prepared)
        report["cases"].append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    numeric_pass = bool(boundary["pass"]) and all(
        item["status"] == "numeric_pass" for item in report["cases"]
    )
    report["numeric_gate_summary"] = {
        "boundary_pass": bool(boundary["pass"]),
        "all_cases_pass": numeric_pass,
        "gate": "finite + input contract + upstream-derived tolerance",
    }
    if numeric_pass:
        for index, prepared in enumerate(prepared_cases):
            report["cases"][index]["timing"] = _timing_case(
                prepared,
                args.warmups,
                args.samples,
                flush,
                args.seed + index,
            )
        report["speed_summary"] = _speed_summary(report["cases"])
        speed_32k = report["speed_summary"].get("cached32768_q256")
        report["continuation_speed_gate"] = {
            "speedup": speed_32k["speedup"] if speed_32k else None,
            "threshold": SPEED_GATE,
            "pass": bool(speed_32k and speed_32k["gate"]),
        }
        report["status"] = (
            "continuation_speed_gate_passed_model_ab_pending"
            if report["continuation_speed_gate"]["pass"]
            else "continuation_speed_gate_failed"
        )
    else:
        report["status"] = "upstream_numerical_gate_failed"
        report["speed_summary"] = {}
        report["continuation_speed_gate"] = {
            "pass": False,
            "status": "not_measured_until_numeric_prerequisites_pass",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
