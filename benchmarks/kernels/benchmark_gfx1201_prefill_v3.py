# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""R0/R1 benchmark for the gfx1201 prefill rearchitecture.

R0 freezes provenance, environment, effective shape weights, input inventory,
and numerical gates.  R1 then measures only the benchmark-only H1/H2 raw-FP8
mappings against same-session BF16/FP32 library GEMMs and the old A3 diagnostic
kernel.  No MXFP4 decode, scale folding, production registration, dispatch, or
model execution is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

BASE_REVISION = "ba57aece9e60d637459556ace36708a9a025cb26"
HANDOFF_BASE_REVISION = "9938409e924ba0418b65dfae8e56304a070726a6"
FP8_DTYPE = torch.float8_e4m3fn
MODEL_SHAPES = (
    (5120, 6144, 64),
    (34816, 5120, 64),
    (5120, 17408, 64),
    (16384, 5120, 48),
    (96, 5120, 48),
    (14336, 5120, 16),
)
DEFAULT_ROWS = (64, 256)
N96_INDEX = 4
N_GE_512_INDICES = (0, 1, 2, 3, 5)


@dataclass(frozen=True)
class TimingConfig:
    warmups: int = 5
    samples: int = 20
    flush_mib: int = 64


VARIANTS = {
    "old_a3": "historical 4-wave 64x128 K16 diagnostic mapping",
    "h1": "R1 H1 128x64 K64, four waves, FP32 direct scratch plus BF16 postcast",
    "h2": "R1 H2 256x64 K64, eight waves, FP32 direct scratch plus BF16 postcast",
    "h2_direct_bf16": "R1 H2 with direct BF16 epilogue and no global FP32 scratch",
    "torch_bf16_mm": "same-session pre-expanded BF16 torch.mm",
    "torch_fp32_mm": "same-session pre-expanded FP32 torch.mm control",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_revision() -> str | None:
    repository = _repository_root()
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


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_fp8_cpu(shape: tuple[int, int], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randn(shape, dtype=torch.float32, generator=generator)
    values = (values * 0.25).clamp(-4.0, 4.0)
    return values.to(FP8_DTYPE).view(torch.uint8).contiguous()


def _input_inventory() -> list[dict[str, object]]:
    inventory = []
    for name, shape, seed in (
        ("r1_random_129x129x64", (129, 64), 9101),
        ("r1_random_65x129x65_a", (65, 65), 9102),
        ("r1_random_65x129x65_b", (129, 65), 9103),
        ("r1_shape0_a", (256, 6144), 9104),
        ("r1_shape0_b", (5120, 6144), 9105),
    ):
        raw = _raw_fp8_cpu(shape, seed)
        inventory.append(
            {
                "name": name,
                "shape": list(shape),
                "seed": seed,
                "dtype": "uint8 raw FP8 E4M3FN bytes",
                "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
            }
        )
    return inventory


def _find_header(name: str) -> dict[str, object]:
    roots: list[Path] = []
    for variable in ("CPATH", "CPLUS_INCLUDE_PATH"):
        roots.extend(
            Path(value) for value in os.environ.get(variable, "").split(":") if value
        )
    roots.extend(
        Path(value)
        for value in (
            "/opt/rocm/include",
            "/opt/rocm/core-7.14/include",
            "/tmp/tq-rocmcore/_rocm_sdk_core/include",
            "/tmp/tq-p3-v2a-sdk-root/include",
        )
    )
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            return {
                "path": str(candidate),
                "sha256": _sha256_file(candidate),
            }
    return {"path": None, "sha256": None}


def _environment_manifest() -> dict[str, object]:
    gpu: dict[str, object] = {"available": False}
    if torch.accelerator.is_available() and torch.version.hip is not None:
        properties = torch.cuda.get_device_properties()
        gpu = {
            "available": True,
            "name": properties.name,
            "gcn_arch": properties.gcnArchName,
            "device": str(torch.accelerator.current_accelerator()),
            "total_memory": properties.total_memory,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": gpu,
        "hipcc": shutil.which("hipcc"),
        "hipcc_version": _command_output(["hipcc", "--version"]),
        "llvm_objdump": shutil.which("llvm-objdump"),
        "rocm_include": {
            "hip_runtime": _find_header("hip/hip_runtime.h"),
            "rocwmma": _find_header("rocwmma/rocwmma.hpp"),
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ROCM_PATH",
                "ROCM_HOME",
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "PYTORCH_ROCM_ARCH",
            )
        },
    }


def describe_plan() -> dict[str, object]:
    return {
        "status": "R0_manifest_then_R1_benchmark_only",
        "base_revision": BASE_REVISION,
        "handoff_declared_base": HANDOFF_BASE_REVISION,
        "plan": "docs/design/gfx1201_prefill_rearchitecture.md",
        "shapes_n_k_calls": [list(shape) for shape in MODEL_SHAPES],
        "rows": list(DEFAULT_ROWS),
        "primary_rows": [256],
        "primary_shape_indices": list(N_GE_512_INDICES),
        "n96": "fallback_only",
        "variants": VARIANTS,
        "r1_gate": {
            "correctness_required": True,
            "weighted_speedup_vs_same_session_bf16": ">= 1.0x",
            "weighted_scope": "M=256, N>=512, calls-weighted median latency",
            "historical_a0_5x_gate_reused": False,
        },
    }


def run_reference_manifest(output: Path) -> None:
    manifest = {
        "stage": "R0",
        "status": "complete_manifest_frozen",
        "revision": _git_revision(),
        "requested_base_revision": BASE_REVISION,
        "handoff_declared_base_revision": HANDOFF_BASE_REVISION,
        "plan": "docs/design/gfx1201_prefill_rearchitecture.md",
        "environment": _environment_manifest(),
        "provenance": {
            "classification": "source-informed, not source-blind clean-room",
            "radiance_image_digest": (
                "sha256:83a9dc02a8f8e75aabe81366d36ebaa2e35fcbe181cacf8e8e0a4cef4ebccbcc"
            ),
            "radiance_source_label": "f295b9ef51ad413a68e4192371e0377741a354ce",
            "observed_sources": [
                "magiccodingman/vllm-radiance/radiance_mxfp4_fp8.hip",
                "magiccodingman/vllm-radiance/radiance_mxfp4.py",
                "magiccodingman/vllm-radiance/radiance_r4d_attn.py",
            ],
            "reuse_status": "no implementation text or lookup table copied",
            "independent_basis": [
                "declared raw-FP8 tensor contract",
                "official rocWMMA interfaces",
                "local gfx1201 compiler/header installation",
            ],
        },
        "preserved_lanes": {
            "decode_mtp2_k8v4": "unchanged",
            "graph_prefix_reuse_workspace": "unchanged",
            "full_decode_only_compilation_disabled": "unchanged",
            "old_p2_p3_p4_candidates": "diagnostic references only",
            "r2_mxfp4_fusion": "not started",
            "production_registration": "off",
        },
        "effective_shape_contract": {
            "dtype_input": "raw FP8 E4M3FN bytes",
            "dtype_output": "float32 diagnostic, BF16 cast comparison",
            "layout_a": "[M,K] contiguous",
            "layout_b": "[N,K] contiguous, result A @ B.T",
            "k_slab": 64,
            "eligible_m": [64, 256],
            "eligible_n_min": 512,
            "fallback_n": [96],
            "k_divisible_by": 64,
            "shape_calls": [list(shape) for shape in MODEL_SHAPES],
            "call_weight_formula": "sum(calls * median_us)",
        },
        "input_inventory": _input_inventory(),
        "numerical_contract": {
            "oracle": "decode identical raw FP8 bytes to FP64 and accumulate in FP64",
            "precast_limits": {
                "finite": True,
                "normalized_max_abs": "<= 1e-3",
                "relative_l2": "<= 1e-4",
            },
            "final_bf16": (
                "compare candidate FP32 cast to BF16 against FP64 reference cast "
                "to BF16; "
                "allow 1.10x cast-only error plus the precast allowance"
            ),
            "record": [
                "max_abs",
                "normalized_max_abs",
                "rmse",
                "relative_l2",
                "BF16 mismatch fraction",
                "finite",
            ],
            "zero_reference": "exact zero is required",
            "old_attention_absolute_threshold_reused": False,
        },
        "r0_coverage_status": {
            "synthetic_gemm_manifest": "frozen_and_measured",
            "real_model_shape_phase_role_calls": "not_collected",
            "matched_fork_radiance_control": "not_run",
        },
        "baseline_contract": {
            "primary": "same-session pre-expanded BF16 torch.mm",
            "secondary": "same-session FP32 torch.mm diagnostic",
            "history": "old A3 raw-FP8 WMMA only",
            "production_full_emulation": (
                "recorded as the future R2 baseline; not silently substituted in R1"
            ),
        },
        "timing_contract": {
            "warmups": 5,
            "samples": 20,
            "order": "rotating operation order per sample",
            "cache_flush_mib": 64,
            "excluded": [
                "allocation",
                "input generation",
                "FP8-to-BF16/FP32 conversion",
                "compilation",
                "oracle and correctness",
            ],
        },
        "r1_hypotheses": {
            "H1": {
                "tile": [128, 64],
                "k_slab": 64,
                "waves": 4,
                "mapping": "four wave32 rows, each wave owns 32x64 output",
            },
            "H2": {
                "tile": [256, 64],
                "k_slab": 64,
                "waves": 8,
                "mapping": "eight wave32 rows, each wave owns 32x64 output",
            },
            "shared_requirements": [
                "aligned 16-byte global loads where valid",
                "safe zero-filled M/N/K edges",
                "padded LDS A/B strides",
                "register accumulators",
                "direct global stores on complete tiles",
                "no full output-tile LDS accumulator",
            ],
        },
        "gate_policy": {
            "r1_pass": (
                "candidate correctness complete and calls-weighted M=256/N>=512 "
                "median latency is no slower than same-session BF16"
            ),
            "r1_failure": "stop/report; do not begin R2 automatically",
            "a0_5x": "historical P3-v2 gate, not reused",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps({"stage": "R0", "output": str(output), "status": manifest["status"]})
    )


def _load_extension(
    *, name: str, source: Path, build_directory: Path, verbose: bool
) -> Any:
    from torch.utils.cpp_extension import load

    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name=name,
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--offload-arch=gfx1201"],
        with_cuda=True,
        verbose=verbose,
    )


def _fp8_bytes(shape: tuple[int, int], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randn(shape, dtype=torch.float32, generator=generator)
    values = (values * 0.25).clamp(-4.0, 4.0)
    return values.to(FP8_DTYPE).view(torch.uint8).to("cuda")


def _as_fp32(raw: torch.Tensor) -> torch.Tensor:
    return raw.view(FP8_DTYPE).float()


def _metric(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    actual64 = actual.double()
    reference64 = reference.double()
    delta = actual64 - reference64
    reference_norm = reference64.norm().item()
    reference_scale = max(reference64.abs().max().item(), 1e-30)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    return {
        "finite": finite,
        "max_abs": delta.abs().max().item(),
        "normalized_max_abs": delta.abs().max().item() / reference_scale,
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": delta.norm().item() / max(reference_norm, 1e-30),
    }


def _bf16_metrics(actual: torch.Tensor, reference64: torch.Tensor) -> dict[str, object]:
    actual_bf16 = actual.to(torch.bfloat16).float()
    reference_bf16 = reference64.to(torch.bfloat16).float()
    cast_only = _metric(reference_bf16, reference64)
    candidate = _metric(actual_bf16, reference64)
    mismatch = (
        actual_bf16.view(torch.int32) != reference_bf16.view(torch.int32)
    ).float()
    return {
        "candidate_bf16_vs_fp64": candidate,
        "cast_only_fp64_to_bf16": cast_only,
        "mismatch_fraction": mismatch.mean().item(),
        "allowed": {
            "max_abs": 1.10 * float(cast_only["max_abs"])
            + 1e-3 * max(reference64.abs().max().item(), 1e-30),
            "relative_l2": 1.10 * float(cast_only["relative_l2"]) + 1e-4,
        },
    }


def _representative_indices(size: int) -> list[int]:
    return sorted(
        {
            index
            for index in (0, 1, 15, 16, 31, 32, 63, 64, size - 2, size - 1)
            if 0 <= index < size
        }
    )


def _oracle_slice(
    a: torch.Tensor,
    b: torch.Tensor,
    row_indices: Iterable[int],
    col_indices: Iterable[int],
) -> torch.Tensor:
    rows = torch.tensor(list(row_indices), device="cuda", dtype=torch.long)
    cols = torch.tensor(list(col_indices), device="cuda", dtype=torch.long)
    a64 = a.index_select(0, rows).view(FP8_DTYPE).double()
    b64 = b.index_select(0, cols).view(FP8_DTYPE).double()
    return torch.mm(a64, b64.t())


def _invoke(
    extension: Any, method: str, a: torch.Tensor, b: torch.Tensor, output: torch.Tensor
) -> None:
    getattr(extension, method)(a, b, output)


def _invoke_bf16(
    extension: Any,
    method: str,
    a: torch.Tensor,
    b: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
) -> None:
    getattr(extension, method)(a, b, scratch, output)


def _correctness_case(
    old_extension: Any,
    new_extension: Any,
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    scope: str,
) -> dict[str, object]:
    m, k = a.shape
    n = b.shape[0]
    outputs = {
        variant: torch.empty((m, n), device="cuda", dtype=torch.float32)
        for variant in ("old_a3", "h1", "h2")
    }
    bf16_outputs = {
        variant: torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        for variant in ("h1", "h2")
    }
    direct_bf16_output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    _invoke(old_extension, "fp8_wmma_gemm_4wave_64x128", a, b, outputs["old_a3"])
    _invoke(new_extension, "raw_fp8_h1", a, b, outputs["h1"])
    _invoke(new_extension, "raw_fp8_h2", a, b, outputs["h2"])
    _invoke_bf16(
        new_extension, "raw_fp8_h1_bf16", a, b, outputs["h1"], bf16_outputs["h1"]
    )
    _invoke_bf16(
        new_extension, "raw_fp8_h2_bf16", a, b, outputs["h2"], bf16_outputs["h2"]
    )
    _invoke(
        new_extension,
        "raw_fp8_h2_bf16_direct",
        a,
        b,
        direct_bf16_output,
    )
    torch.accelerator.synchronize()
    if scope == "full":
        row_indices = list(range(m))
        col_indices = list(range(n))
    else:
        row_indices = _representative_indices(m)
        col_indices = _representative_indices(n)
    oracle = _oracle_slice(a, b, row_indices, col_indices)
    records: dict[str, object] = {
        "case": name,
        "shape": [m, n, k],
        "scope": scope,
        "oracle": "identical raw FP8 bytes decoded to FP64 with FP64 accumulation",
        "slice": [len(row_indices), len(col_indices)],
        "variants": {},
        "timed_output_dtype": (
            "bfloat16 for H1/H2/direct H2; float32 retained as a diagnostic"
        ),
    }
    variant_records = records["variants"]
    assert isinstance(variant_records, dict)
    for variant, output in outputs.items():
        row_tensor = torch.tensor(row_indices, device="cuda", dtype=torch.long)
        col_tensor = torch.tensor(col_indices, device="cuda", dtype=torch.long)
        actual = output.index_select(0, row_tensor).index_select(1, col_tensor)
        precast = _metric(actual, oracle)
        if variant in bf16_outputs:
            bf16_actual = (
                bf16_outputs[variant]
                .index_select(0, row_tensor)
                .index_select(1, col_tensor)
            )
        else:
            bf16_actual = actual
        bf16 = _bf16_metrics(bf16_actual, oracle)
        finite_full = bool(torch.isfinite(output).all())
        if variant in bf16_outputs:
            finite_full = finite_full and bool(
                torch.isfinite(bf16_outputs[variant]).all()
            )
        passed = bool(
            finite_full
            and bool(precast["finite"])
            and float(precast["normalized_max_abs"]) <= 1e-3
            and float(precast["relative_l2"]) <= 1e-4
            and float(bf16["candidate_bf16_vs_fp64"]["max_abs"])
            <= float(bf16["allowed"]["max_abs"])
            and float(bf16["candidate_bf16_vs_fp64"]["relative_l2"])
            <= float(bf16["allowed"]["relative_l2"])
        )
        variant_records[variant] = {
            "finite_full_output": finite_full,
            "precast": precast,
            "final_bf16": bf16,
            "passed": passed,
        }

    direct_rows = direct_bf16_output.index_select(0, row_tensor)
    direct_actual = direct_rows.index_select(1, col_tensor)
    direct_bf16 = _bf16_metrics(direct_actual.float(), oracle)
    direct_equal = bool(torch.equal(direct_bf16_output, bf16_outputs["h2"]))
    direct_finite = bool(torch.isfinite(direct_bf16_output).all())
    direct_passed = bool(
        direct_finite
        and direct_equal
        and float(direct_bf16["candidate_bf16_vs_fp64"]["max_abs"])
        <= float(direct_bf16["allowed"]["max_abs"])
        and float(direct_bf16["candidate_bf16_vs_fp64"]["relative_l2"])
        <= float(direct_bf16["allowed"]["relative_l2"])
    )
    variant_records["h2_direct_bf16"] = {
        "finite_full_output": direct_finite,
        "precast": {"available": False, "reason": "direct BF16 output only"},
        "final_bf16": direct_bf16,
        "bitwise_equal_to_h2_postcast": direct_equal,
        "passed": direct_passed,
    }
    records["passed"] = all(
        bool(value["passed"])
        for value in variant_records.values()
        if isinstance(value, dict)
    )
    return records


def _make_basis_case() -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.zeros((2, 64), device="cpu", dtype=torch.float32)
    b = torch.zeros((3, 64), device="cpu", dtype=torch.float32)
    a[0, 0] = 1.0
    a[0, 63] = 2.0
    a[1, 16] = -1.5
    a[1, 47] = 0.75
    b[0, 0] = 0.5
    b[0, 16] = 2.0
    b[1, 47] = -3.0
    b[2, 63] = 1.25
    return a.to(FP8_DTYPE).view(torch.uint8).to("cuda"), b.to(FP8_DTYPE).view(
        torch.uint8
    ).to("cuda")


def _make_distinct_case(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(m, dtype=torch.float32)[:, None]
    cols = torch.arange(n, dtype=torch.float32)[:, None]
    kvals = torch.arange(k, dtype=torch.float32)[None, :]
    a = ((rows % 17 - 8) * 0.03125 + (kvals % 7 - 3) * 0.0078125).clamp(-4, 4)
    b = ((cols % 19 - 9) * 0.02734375 - (kvals % 11 - 5) * 0.005859375).clamp(-4, 4)
    return a.to(FP8_DTYPE).view(torch.uint8).to("cuda"), b.to(FP8_DTYPE).view(
        torch.uint8
    ).to("cuda")


def _correctness_inventory(
    old_extension: Any, new_extension: Any
) -> list[dict[str, object]]:
    cases: list[tuple[str, torch.Tensor, torch.Tensor, str]] = []
    basis_a, basis_b = _make_basis_case()
    cases.append(("basis_vectors", basis_a, basis_b, "full"))
    distinct_a, distinct_b = _make_distinct_case(129, 129, 64)
    cases.append(("row_column_distinct", distinct_a, distinct_b, "full"))
    random_a = _fp8_bytes((65, 65), 9102)
    random_b = _fp8_bytes((129, 65), 9103)
    cases.append(("mnk_tails", random_a, random_b, "full"))
    for name, m, n, k, seed in (
        ("h1_full_tile", 128, 64, 64, 9111),
        ("h2_full_tile", 256, 64, 64, 9112),
    ):
        cases.append(
            (name, _fp8_bytes((m, k), seed), _fp8_bytes((n, k), seed + 1), "full")
        )
    for shape_index, (n, k, _calls) in enumerate(MODEL_SHAPES):
        for m in DEFAULT_ROWS:
            seed = 9200 + shape_index * 100 + m
            cases.append(
                (
                    f"shape{shape_index}_m{m}",
                    _fp8_bytes((m, k), seed),
                    _fp8_bytes((n, k), seed + 1),
                    "representative",
                )
            )
    results = []
    for name, a, b, scope in cases:
        results.append(
            _correctness_case(old_extension, new_extension, name, a, b, scope)
        )
    return results


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
            timings[name].append(start.elapsed_time(end) * 1000.0)
    return timings


def _timing_case(
    old_extension: Any,
    new_extension: Any,
    m: int,
    n: int,
    k: int,
    calls: int,
    flush: torch.Tensor,
    config: TimingConfig,
    seed: int,
    correctness: dict[str, object] | None,
) -> dict[str, object]:
    a = _fp8_bytes((m, k), seed)
    b = _fp8_bytes((n, k), seed + 1)
    a_fp32 = _as_fp32(a)
    b_fp32 = _as_fp32(b)
    a_bf16 = a_fp32.to(torch.bfloat16)
    b_bf16 = b_fp32.to(torch.bfloat16)
    outputs = {
        name: torch.empty((m, n), device="cuda", dtype=torch.float32)
        for name in ("old_a3", "h1", "h2", "torch_fp32_mm")
    }
    bf16_output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    bf16_outputs = {
        name: torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        for name in ("h1", "h2")
    }
    direct_bf16_output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    operations: dict[str, Callable[[], object]] = {
        "old_a3": lambda: _invoke(
            old_extension, "fp8_wmma_gemm_4wave_64x128", a, b, outputs["old_a3"]
        ),
        "h1": lambda: _invoke_bf16(
            new_extension,
            "raw_fp8_h1_bf16",
            a,
            b,
            outputs["h1"],
            bf16_outputs["h1"],
        ),
        "h2": lambda: _invoke_bf16(
            new_extension,
            "raw_fp8_h2_bf16",
            a,
            b,
            outputs["h2"],
            bf16_outputs["h2"],
        ),
        "h2_direct_bf16": lambda: _invoke(
            new_extension,
            "raw_fp8_h2_bf16_direct",
            a,
            b,
            direct_bf16_output,
        ),
        "torch_bf16_mm": lambda: torch.mm(a_bf16, b_bf16.t(), out=bf16_output),
        "torch_fp32_mm": lambda: torch.mm(
            a_fp32, b_fp32.t(), out=outputs["torch_fp32_mm"]
        ),
    }
    candidate_scope = "timed"
    if n == 96:
        candidate_scope = "fallback_not_timed_for_large-N_gate"
        operations.pop("h1")
        operations.pop("h2")
        operations.pop("h2_direct_bf16")
    raw = _measure_round_robin(operations, flush, config)
    timing = {name: _summary(values) for name, values in raw.items()}
    flop = 2.0 * m * n * k
    tflops = {
        name: flop / (values["median_us"] * 1e-6) / 1e12
        for name, values in timing.items()
    }
    speedup_vs_bf16 = {
        name: timing["torch_bf16_mm"]["median_us"] / values["median_us"]
        for name, values in timing.items()
        if name in ("old_a3", "h1", "h2", "h2_direct_bf16")
    }
    return {
        "m": m,
        "n": n,
        "k": k,
        "calls": calls,
        "seed": seed,
        "candidate_scope": candidate_scope,
        "timing": timing,
        "effective_tflops": tflops,
        "speedup_vs_same_session_bf16": speedup_vs_bf16,
        "correctness": correctness if correctness is not None else {"skipped": True},
        "output_dtype": {
            "old_a3": "float32",
            "h1": "bfloat16 (FP32 scratch + postcast in timed wrapper)",
            "h2": "bfloat16 (FP32 scratch + postcast in timed wrapper)",
            "h2_direct_bf16": "bfloat16 (direct epilogue, no FP32 scratch)",
            "torch_bf16_mm": "bfloat16",
            "torch_fp32_mm": "float32",
        },
        "inputs": {
            "a": "raw FP8 E4M3FN bytes, preexpanded",
            "b": "raw FP8 E4M3FN bytes, preexpanded [N,K]",
            "decode": False,
            "scales": False,
        },
        "timed_region": (
            "one fixed-buffer operation; allocation, input generation, conversion, "
            "compilation, oracle and correctness are excluded"
        ),
    }


def _candidate_correctness_complete(correctness: list[dict[str, object]]) -> bool:
    return bool(correctness) and all(bool(case.get("passed")) for case in correctness)


def _weighted_gate(
    records: list[dict[str, object]], correctness: list[dict[str, object]]
) -> dict[str, object]:
    selected = [
        record for record in records if record["m"] == 256 and record["n"] >= 512
    ]
    expected_shape_ids = set(N_GE_512_INDICES)
    expected = len(expected_shape_ids)
    observed_shape_ids = [record.get("shape_index") for record in selected]
    unique_shape_ids = {
        shape_id for shape_id in observed_shape_ids if isinstance(shape_id, int)
    }
    shape_id_set_complete = (
        len(selected) == expected
        and len(unique_shape_ids) == expected
        and unique_shape_ids == expected_shape_ids
    )
    selected_by_shape = {
        record["shape_index"]: record
        for record in selected
        if isinstance(record.get("shape_index"), int)
    }
    gate: dict[str, object] = {
        "scope": "M=256, N>=512, calls-weighted median latency",
        "baseline": "same-session torch_bf16_mm",
        "threshold": 1.0,
        "expected_shape_count": expected,
        "completed_shape_count": len(selected),
        "unique_shape_count": len(unique_shape_ids),
        "expected_shape_ids": sorted(expected_shape_ids),
        "observed_shape_ids": sorted(unique_shape_ids),
        "shape_id_set_complete": shape_id_set_complete,
        "correctness_complete": _candidate_correctness_complete(correctness),
        "candidates": {},
    }
    candidates = gate["candidates"]
    assert isinstance(candidates, dict)
    for candidate in ("old_a3", "h1", "h2", "h2_direct_bf16"):
        candidate_records = [
            record
            for record in selected_by_shape.values()
            if candidate in record["timing"]
        ]
        if not shape_id_set_complete or len(candidate_records) != expected:
            candidates[candidate] = {
                "status": "incomplete",
                "passed": False,
                "weighted_speedup": None,
            }
            continue
        baseline_total = sum(
            int(record["calls"]) * float(record["timing"]["torch_bf16_mm"]["median_us"])
            for record in candidate_records
        )
        candidate_total = sum(
            int(record["calls"]) * float(record["timing"][candidate]["median_us"])
            for record in candidate_records
        )
        speedup = baseline_total / candidate_total
        passed = bool(
            gate["correctness_complete"] and speedup >= float(gate["threshold"])
        )
        candidates[candidate] = {
            "weighted_baseline_us": baseline_total,
            "weighted_candidate_us": candidate_total,
            "weighted_speedup_vs_bf16": speedup,
            "passed": passed,
        }
    h_passed = [
        name
        for name in ("h1", "h2", "h2_direct_bf16")
        if candidates[name].get("passed")
    ]
    gate["selected_candidates"] = h_passed
    gate["decision"] = (
        "R1 pass; repeat selected mapping validation before proposing R2"
        if h_passed
        else "R1 stop; no H1/H2 variant met the same-session BF16 feasibility gate"
    )
    gate["r2_started"] = False
    return gate


def _metadata(
    args: argparse.Namespace, manifest: dict[str, object]
) -> dict[str, object]:
    properties = torch.cuda.get_device_properties()
    return {
        "stage": "R1",
        "revision": _git_revision(),
        "requested_base_revision": BASE_REVISION,
        "manifest_revision": manifest.get("revision"),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(torch.accelerator.current_accelerator()),
        "device_name": properties.name,
        "gcn_arch": properties.gcnArchName,
        "timing_config": asdict(
            TimingConfig(args.warmups, args.samples, args.flush_mib)
        ),
        "rows": args.rows,
        "shape_indices": args.shape_index,
        "build_directory": str(args.build_directory),
        "source": "csrc/rocm/gfx1201_prefill_v3.cu",
        "old_source": "benchmarks/kernels/gfx1201_fp8_wmma_microbenchmark.cu",
        "variants": VARIANTS,
        "timed_output_contract": {
            "h1": (
                "BF16 output from preallocated FP32 direct-output scratch plus postcast"
            ),
            "h2": (
                "BF16 output from preallocated FP32 direct-output scratch plus postcast"
            ),
            "h2_direct_bf16": (
                "BF16 output from direct BF16 epilogue without FP32 scratch"
            ),
            "torch_bf16_mm": "BF16 output",
            "fp32_diagnostic": "separate, not the primary gate",
        },
        "scope": (
            "benchmark-only raw FP8 mapping; no MXFP4 decode, E8M0 scale, "
            "production registration, threshold, or model integration"
        ),
        "mapping_gate": {
            "baseline": "same-session torch_bf16_mm",
            "threshold": 1.0,
            "weighted_scope": "M=256, N>=512, calls-weighted median latency",
            "correctness_required": True,
            "historical_a0_gate_reused": False,
        },
    }


def run_mapping_benchmark(
    manifest_path: Path, output: Path, args: argparse.Namespace
) -> None:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("stage") != "R0"
        or manifest.get("status") != "complete_manifest_frozen"
    ):
        raise ValueError("R1 requires a completed R0 manifest")
    if not torch.accelerator.is_available() or torch.version.hip is None:
        raise RuntimeError("R1 requires a ROCm GPU")
    properties = torch.cuda.get_device_properties()
    if properties.gcnArchName != "gfx1201":
        raise RuntimeError(f"R1 requires gfx1201, got {properties.gcnArchName}")
    if args.warmups < 0 or args.samples < 1 or args.flush_mib < 1:
        raise ValueError("warmups/samples/flush_mib must be positive where applicable")

    source = _repository_root() / "csrc/rocm/gfx1201_prefill_v3.cu"
    old_source = (
        _repository_root() / "benchmarks/kernels/gfx1201_fp8_wmma_microbenchmark.cu"
    )
    new_extension = _load_extension(
        name="gfx1201_prefill_v3_r1",
        source=source,
        build_directory=args.build_directory,
        verbose=args.verbose_build,
    )
    old_extension = _load_extension(
        name="gfx1201_fp8_wmma_r0_old_a3",
        source=old_source,
        build_directory=args.build_directory / "old_a3",
        verbose=args.verbose_build,
    )
    config = TimingConfig(args.warmups, args.samples, args.flush_mib)
    flush = torch.empty(args.flush_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)
    correctness = (
        []
        if args.skip_correctness
        else _correctness_inventory(old_extension, new_extension)
    )

    selected = (
        list(range(len(MODEL_SHAPES)))
        if args.shape_index is None
        else list(args.shape_index)
    )
    for index in selected:
        if index < 0 or index >= len(MODEL_SHAPES):
            raise ValueError(f"shape index {index} is out of range")

    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as output_file:
        metadata = _metadata(args, manifest)
        metadata["correctness_inventory"] = correctness
        output_file.write(json.dumps({"metadata": metadata}) + "\n")
        output_file.flush()
        print(json.dumps({"metadata": metadata}), flush=True)
        for shape_index in selected:
            n, k, calls = MODEL_SHAPES[shape_index]
            for m in args.rows:
                record = _timing_case(
                    old_extension,
                    new_extension,
                    m,
                    n,
                    k,
                    calls,
                    flush,
                    config,
                    seed=1201 + shape_index * 10000 + m,
                    correctness=None,
                )
                record["shape_index"] = shape_index
                records.append(record)
                output_file.write(json.dumps(record) + "\n")
                output_file.flush()
                print(json.dumps(record), flush=True)
        gate = _weighted_gate(records, correctness)
        output_file.write(json.dumps({"mapping_gate": gate}) + "\n")
        output_file.flush()
        print(json.dumps({"mapping_gate": gate}), flush=True)
        print(f"saved {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--describe", action="store_true")
    action.add_argument("--stage", choices=("R0", "R1"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shape-index", type=int, nargs="+", default=None)
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument(
        "--build-directory",
        type=Path,
        default=Path("/tmp/tq-gfx1201-prefill-v3-r1-build"),
    )
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe_plan(), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for a stage run")
    if args.stage == "R0":
        run_reference_manifest(args.output)
        return
    if args.manifest is None:
        parser.error("--manifest is required for R1")
    run_mapping_benchmark(args.manifest, args.output, args)


if __name__ == "__main__":
    main()
