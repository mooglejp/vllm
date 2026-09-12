# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native Gemma-style RMSNorm numerics across identical-row batches.

This diagnostic reports errors against FP64, without timing or changing kernels.
An optional snapshot contains x, residual, the unshifted weight, and epsilon.
"""

import argparse
import json
from pathlib import Path

import torch

from vllm import ir


def compare_rows(x, residual, weight, epsilon, row_counts):
    native = ir.ops.fused_add_rms_norm.impls["native"].impl_fn
    weight = weight.float() + 1.0
    total = x.double() + residual.double()
    reference = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + epsilon)
    reference *= weight.double()
    reference = reference.cpu()
    x, residual, weight = (value.to("cuda") for value in (x, residual, weight))
    single = native(x, residual, weight, epsilon)[0].cpu()
    records = []
    for count in row_counts:
        batch_x = x.expand(count, -1).contiguous()
        batch_residual = residual.expand(count, -1).contiguous()
        actual = native(batch_x, batch_residual, weight, epsilon)[0][:1].cpu()
        assert torch.isfinite(actual).all()
        delta = actual.double() - reference
        records.append(
            dict(
                rows=count,
                variance=float(
                    (batch_x.float() + batch_residual.float()).square().mean(-1)[0]
                ),
                changed_elements=int((actual != single).sum()),
                max_abs_vs_single=float((actual.float() - single.float()).abs().max()),
                max_abs_vs_fp64=float(delta.abs().max()),
                rms_vs_fp64=float(delta.square().mean().sqrt()),
            )
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-snapshot", type=Path)
    parser.add_argument("--row", type=int, default=-1)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--row-counts", type=int, nargs="+", default=[1, 2, 3, 4, 8])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(count < 1 for count in args.row_counts):
        parser.error("row counts must be positive")
    if args.input_snapshot:
        bundle = torch.load(args.input_snapshot, map_location="cpu", weights_only=True)
        x = bundle["x"][args.row].unsqueeze(0)
        residual = bundle["residual"][args.row].unsqueeze(0)
        weight, epsilon = bundle["weight"], bundle["epsilon"]
    else:
        torch.manual_seed(args.seed)
        dtype = getattr(torch, args.dtype)
        x = torch.randn(1, args.hidden_size, dtype=dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(args.hidden_size, dtype=dtype)
        epsilon = 1e-6
    result = dict(
        torch_version=str(torch.__version__),
        hip_version=torch.version.hip,
        dtype=str(x.dtype),
        hidden_size=x.shape[-1],
        epsilon=epsilon,
        source=str(args.input_snapshot) if args.input_snapshot else f"seed:{args.seed}",
        records=compare_rows(x, residual, weight, epsilon, args.row_counts),
    )
    encoded = json.dumps(result)
    if args.output:
        with args.output.open("x") as output:
            output.write(encoded + "\n")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
