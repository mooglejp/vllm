# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inactive R0/R1 measurement entry point; --describe is safe on CPU.

See docs/design/gfx1201_prefill_rearchitecture.md. No placeholder run may emit
PASS, a fabricated timing, or silently invoke a rejected candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_REVISION = "9938409e924ba0418b65dfae8e56304a070726a6"
MODEL_SHAPES = (
    (5120, 6144, 64),
    (34816, 5120, 64),
    (5120, 17408, 64),
    (16384, 5120, 48),
    (96, 5120, 48),
    (14336, 5120, 16),
)


def describe_plan() -> dict[str, object]:
    return {
        "status": "scaffold_only_not_measured",
        "base_revision": BASE_REVISION,
        "plan": "docs/design/gfx1201_prefill_rearchitecture.md",
        "initial_shapes_n_k_calls": MODEL_SHAPES,
        "initial_rows": [64, 256],
        "primary_rows": 256,
        "primary_min_n": 512,
        "r1_primary_baseline": "same_session_preexpanded_bf16_gemm",
        "r1_min_weighted_speedup": 1.0,
        "r2_primary_baseline": "full_emulation_qdq_dequant_linear",
        "r2_min_weighted_speedup": 1.25,
        "performance_acceptance": False,
    }


def run_reference_manifest(output: Path) -> None:
    """R0: freeze source/build/shape/input hashes and numeric rules.

    Reuse saved cases where their contracts match. Identify actual GEMM M
    separately from request q_len. Keep profiled sums apart from unprofiled
    latency. The manifest precedes all new candidate measurements.
    """
    raise NotImplementedError("R0 reference-manifest runner is not implemented")


def run_mapping_benchmark(manifest: Path, output: Path) -> None:
    """R1: H1/H2 plus same-session BF16/FP32, fixed buffers and raw samples.

    Freeze shape-specific selection on calibration, validate on another timing
    pass, report missing/unsupported/failed cases, and reject skipped checks.
    Historical A0 ratios are informational and never decide this gate.
    """
    raise NotImplementedError("R1 mapping-benchmark runner is not implemented")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--describe", action="store_true")
    action.add_argument("--stage", choices=("R0", "R1"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe_plan(), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for a stage run")
    if args.stage == "R0":
        run_reference_manifest(args.output)
    else:
        if args.manifest is None:
            parser.error("--manifest is required for R1")
        run_mapping_benchmark(args.manifest, args.output)


if __name__ == "__main__":
    main()
