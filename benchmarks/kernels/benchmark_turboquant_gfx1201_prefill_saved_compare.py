# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare saved P2.2 attention outputs against one shared FP64 reference.

This is a CPU-only diagnostic over immutable ``torch.save`` artifacts. It does
not launch a kernel, modify production dispatch, or feed candidate output into
a baseline calculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def error_metrics(
    actual: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | bool]:
    actual64 = actual.detach().double()
    reference64 = reference.detach().double()
    delta = actual64 - reference64
    return {
        "finite": bool(
            torch.isfinite(actual64).all() and torch.isfinite(reference64).all()
        ),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_l2": float(delta.norm() / reference64.norm().clamp_min(1e-30)),
        "equal_fraction": float((actual == reference).double().mean()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual64.reshape(1, -1),
                reference64.reshape(1, -1),
                dim=1,
            )[0]
        ),
    }


def load(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a tensor mapping")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--pvfp32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--revision", default="8fb487f41266db2e9ba634632dc3cf99e26d8704"
    )
    args = parser.parse_args()

    bf16 = load(args.bf16)
    pvfp32 = load(args.pvfp32)
    required = {"old_sdpa_math", "candidate_streaming", "fp64_reference"}
    for name, tensors in (("bf16", bf16), ("pvfp32", pvfp32)):
        missing = required - tensors.keys()
        if missing:
            raise ValueError(f"{name} artifact is missing {sorted(missing)}")
    if not torch.equal(bf16["fp64_reference"], pvfp32["fp64_reference"]):
        raise ValueError("the two artifacts do not share the same FP64 reference")
    if not torch.equal(bf16["old_sdpa_math"], pvfp32["old_sdpa_math"]):
        raise ValueError("the old SDPA tensors differ between saved artifacts")

    reference = bf16["fp64_reference"]
    records = {
        "old_sdpa_math": error_metrics(bf16["old_sdpa_math"], reference),
        "candidate_bf16_pv": error_metrics(bf16["candidate_streaming"], reference),
        "candidate_pv_fp32": error_metrics(pvfp32["candidate_streaming"], reference),
    }
    result = {
        "revision": args.revision,
        "bf16_artifact": str(args.bf16),
        "pvfp32_artifact": str(args.pvfp32),
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "shared_reference": True,
        "old_sdpa_replay_equal": True,
        "vs_fp64_reference": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
