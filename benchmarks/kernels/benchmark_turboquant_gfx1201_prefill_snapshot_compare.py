# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate FP64 attention arithmetic from BF16 input and output rounding.

The input is one baseline P2.2 snapshot. Prefix K/V, scale, and the causal
boundary are loaded from that snapshot; no production code or kernel is run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

REQUIRED = {
    "query",
    "key_chunk",
    "value_chunk",
    "prefix_key_fp16",
    "prefix_value_fp16",
    "prefix_key_bf16",
    "prefix_value_bf16",
    "q_positions",
    "k_positions",
    "causal_mask",
    "scale",
    "attention_output",
}
TILE_SIZE = 16


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    a = actual.detach().double()
    b = reference.detach().double()
    delta = a - b
    return {
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_l2": float(delta.norm() / b.norm().clamp_min(1e-30)),
        "equal_fraction": float((actual == reference).double().mean()),
    }


def load(path: Path, device: torch.device) -> dict[str, Any]:
    value = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a snapshot mapping")
    missing = REQUIRED - value.keys()
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    if value.get("sdpa_backend") != "MATH" or not value.get("sdpa_math_forced"):
        raise ValueError("snapshot was not captured with explicitly forced Math SDPA")
    return value


def validate_snapshot(saved: dict[str, Any]) -> tuple[int, int, int]:
    query = saved["query"]
    key_chunk = saved["key_chunk"]
    value_chunk = saved["value_chunk"]
    prefix_key_fp16 = saved["prefix_key_fp16"]
    prefix_value_fp16 = saved["prefix_value_fp16"]
    prefix_key_bf16 = saved["prefix_key_bf16"]
    prefix_value_bf16 = saved["prefix_value_bf16"]
    cached_len = int(saved["cached_len"])
    seq_len = int(saved["seq_len"])
    q_len, num_query_heads, head_dim = query.shape
    if key_chunk.shape != value_chunk.shape:
        raise ValueError("raw K/V shapes differ")
    if key_chunk.shape != (q_len, prefix_key_fp16.shape[1], head_dim):
        raise ValueError("raw K/V shape does not match query and prefix")
    expected_prefix = (cached_len, key_chunk.shape[1], head_dim)
    for name, tensor in (
        ("prefix_key_fp16", prefix_key_fp16),
        ("prefix_value_fp16", prefix_value_fp16),
        ("prefix_key_bf16", prefix_key_bf16),
        ("prefix_value_bf16", prefix_value_bf16),
    ):
        if tuple(tensor.shape) != expected_prefix:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {expected_prefix}")
    if seq_len != cached_len + q_len:
        raise ValueError("seq_len is not cached_len + q_len")
    q_positions = saved["q_positions"]
    k_positions = saved["k_positions"]
    causal_mask = saved["causal_mask"]
    if not torch.equal(
        q_positions,
        torch.arange(
            cached_len,
            seq_len,
            dtype=q_positions.dtype,
            device=q_positions.device,
        ),
    ):
        raise ValueError("q_positions do not match the saved continuation boundary")
    if not torch.equal(
        k_positions,
        torch.arange(seq_len, dtype=k_positions.dtype, device=k_positions.device),
    ):
        raise ValueError("k_positions do not match the saved sequence boundary")
    expected_mask = k_positions[None, :] <= q_positions[:, None]
    if not torch.equal(causal_mask, expected_mask):
        raise ValueError("causal_mask does not match q_positions/k_positions")
    if num_query_heads % key_chunk.shape[1] != 0:
        raise ValueError("query/KV head grouping is not integral")
    return int(q_len), int(key_chunk.shape[1]), int(head_dim)


def assemble(prefix: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    return torch.cat((prefix, current), dim=0)


def fp64_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    q_len, num_query_heads, head_dim = query.shape
    num_kv_heads = key.shape[1]
    group = num_query_heads // num_kv_heads
    q = (
        query.to(torch.float64)
        .reshape(q_len, num_kv_heads, group, head_dim)
        .permute(1, 0, 2, 3)
    )
    k = key.to(torch.float64).permute(1, 0, 2)
    v = value.to(torch.float64).permute(1, 0, 2)
    row_max = torch.full(
        (num_kv_heads, q_len, group),
        -float("inf"),
        dtype=torch.float64,
        device=query.device,
    )
    for start in range(0, key.shape[0], TILE_SIZE):
        end = min(start + TILE_SIZE, key.shape[0])
        scores = torch.matmul(q, k[:, start:end].transpose(-1, -2)[:, None]) * scale
        visible = causal_mask[:, start:end].to(device=query.device)
        scores = scores.masked_fill(~visible[None, :, None, :], -float("inf"))
        row_max = torch.maximum(row_max, scores.amax(dim=-1))
    row_sum = torch.zeros_like(row_max)
    output = torch.zeros(
        num_kv_heads,
        q_len,
        group,
        head_dim,
        dtype=torch.float64,
        device=query.device,
    )
    for start in range(0, key.shape[0], TILE_SIZE):
        end = min(start + TILE_SIZE, key.shape[0])
        scores = torch.matmul(q, k[:, start:end].transpose(-1, -2)[:, None]) * scale
        visible = causal_mask[:, start:end].to(device=query.device)
        scores = scores.masked_fill(~visible[None, :, None, :], -float("inf"))
        weights = torch.exp(scores - row_max[..., None])
        row_sum += weights.sum(dim=-1)
        output += torch.matmul(weights, v[:, start:end][:, None])
    output /= row_sum.clamp_min(1e-300)[..., None]
    return (
        output.permute(1, 0, 2, 3)
        .reshape(q_len, num_query_heads, head_dim)
        .contiguous()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-references", type=Path)
    parser.add_argument(
        "--revision", default="8fb487f41266db2e9ba634632dc3cf99e26d8704"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    saved = load(args.snapshot, device)
    q_len, num_kv_heads, head_dim = validate_snapshot(saved)
    query = saved["query"]
    key_chunk = saved["key_chunk"]
    value_chunk = saved["value_chunk"]
    causal_mask = saved["causal_mask"]
    scale = float(saved["scale"])
    prefix_fp16_key = saved["prefix_key_fp16"]
    prefix_fp16_value = saved["prefix_value_fp16"]
    prefix_bf16_key = saved["prefix_key_bf16"]
    prefix_bf16_value = saved["prefix_value_bf16"]

    key_fp16_contract = assemble(prefix_fp16_key, key_chunk)
    value_fp16_contract = assemble(prefix_fp16_value, value_chunk)
    key_bf16_contract = assemble(prefix_bf16_key, key_chunk)
    value_bf16_contract = assemble(prefix_bf16_value, value_chunk)
    fp64_precise = fp64_attention(
        query,
        key_fp16_contract,
        value_fp16_contract,
        causal_mask,
        scale,
    )
    fp64_bf16_input = fp64_attention(
        query,
        key_bf16_contract,
        value_bf16_contract,
        causal_mask,
        scale,
    )
    fp64_precise_cast = fp64_precise.to(query.dtype)
    fp64_bf16_input_cast = fp64_bf16_input.to(query.dtype)
    actual = saved["attention_output"]

    references = {
        "fp64_precast_prefix_fp16": fp64_precise,
        "fp64_precast_prefix_bf16": fp64_bf16_input,
        "fp64_bf16_cast_prefix_fp16": fp64_precise_cast,
        "fp64_bf16_cast_prefix_bf16": fp64_bf16_input_cast,
    }
    result = {
        "revision": args.revision,
        "snapshot": str(args.snapshot),
        "layer": saved["layer"],
        "device": str(device),
        "shape": [q_len, int(query.shape[1]), head_dim],
        "cached_len": int(saved["cached_len"]),
        "seq_len": int(saved["seq_len"]),
        "scale": scale,
        "sdpa_backend": saved["sdpa_backend"],
        "sdpa_math_forced": bool(saved["sdpa_math_forced"]),
        "causal_mask_fixed": True,
        "prefix_rounding": {
            "workspace_dtype": str(prefix_fp16_key.dtype),
            "query_dtype": str(query.dtype),
            "bf16_prefix_saved": True,
        },
        "actual_vs_reference": {
            name: metrics(actual, reference) for name, reference in references.items()
        },
        "rounding_deltas": {
            "prefix_fp16_input_to_prefix_bf16_input_precast": metrics(
                fp64_precise, fp64_bf16_input
            ),
            "prefix_fp16_input_to_prefix_bf16_input_after_cast": metrics(
                fp64_precise_cast, fp64_bf16_input_cast
            ),
            "final_output_cast_prefix_fp16_input": metrics(
                fp64_precise, fp64_precise_cast
            ),
            "final_output_cast_prefix_bf16_input": metrics(
                fp64_bf16_input, fp64_bf16_input_cast
            ),
        },
        "operation_residual": {
            "old_sdpa_vs_bf16_input_fp64_final_cast": metrics(
                actual, fp64_bf16_input_cast
            ),
            "old_sdpa_vs_bf16_input_fp64_precast": metrics(actual, fp64_bf16_input),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.save_references:
        args.save_references.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "revision": args.revision,
                "snapshot": str(args.snapshot),
                "layer": saved["layer"],
                "sdpa_backend": saved["sdpa_backend"],
                "sdpa_math_forced": bool(saved["sdpa_math_forced"]),
                "fp64_precast_prefix_fp16": fp64_precise.detach().cpu(),
                "fp64_precast_prefix_bf16": fp64_bf16_input.detach().cpu(),
                "fp64_bf16_cast_prefix_fp16": fp64_precise_cast.detach().cpu(),
                "fp64_bf16_cast_prefix_bf16": fp64_bf16_input_cast.detach().cpu(),
                "actual_sdpa_math": actual.detach().cpu(),
            },
            args.save_references,
        )
        result["reference_artifact"] = str(args.save_references)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
