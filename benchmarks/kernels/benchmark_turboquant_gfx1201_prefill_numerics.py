# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay P2.2 raw-current prefill numerics on one immutable model snapshot.

The snapshot is captured from the baseline continuation path.  This diagnostic
never feeds candidate output into a model run and does not modify production
thresholds or dispatch.  It compares the old SDPA path, the streaming kernel,
online-softmax ablations with higher-precision PV or QK+PV, and an FP64
matrix reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.turboquant_soa.gfx1201_prefill import (
    launch_gfx1201_tq_continuation_prefill,
)
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
)
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode_gfx1201_k8v4 import (
    DATA_BYTES_PER_SLOT,
    KEY_DATA_BYTES,
    NUM_SOA_FIELDS,
    SOA_V_SCALE,
    SOA_V_ZERO,
)

TILE_SIZE = 16
HEAD_DIM = 256
NUM_KV_HEADS = 4
NUM_QUERY_HEADS = 24
VALUE_BYTES = 128


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    a = actual.detach().double()
    b = reference.detach().double()
    delta = a - b
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {
        "finite": finite,
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_l2": float(delta.norm() / b.norm().clamp_min(1e-30)),
        "equal_fraction": float((actual == reference).double().mean()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                a.reshape(1, -1), b.reshape(1, -1), dim=1
            )[0]
        ),
    }


def dequant_prefix(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cached_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = int(cache.shape[1])
    alloc_len = math.ceil(cached_len / block_size) * block_size
    key = torch.empty(
        1,
        NUM_KV_HEADS,
        alloc_len,
        HEAD_DIM,
        dtype=torch.float16,
        device=cache.device,
    )
    value = torch.empty_like(key)
    data_bytes = DATA_BYTES_PER_SLOT
    meta_region_offset = block_size * NUM_KV_HEADS * data_bytes
    _tq_full_dequant_kv[(alloc_len, NUM_KV_HEADS)](
        cache,
        cache.view(torch.uint16),
        block_table,
        torch.empty(1, dtype=torch.float32, device=cache.device),
        key,
        value,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        cache.stride(0),
        block_table.stride(0),
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=NUM_KV_HEADS,
        MSE_BYTES=0,
        VQB=4,
        VAL_DATA_BYTES=VALUE_BYTES,
        MSE_BITS=0,
        KEY_FP8=1,
        KEY_DATA_BYTES=KEY_DATA_BYTES,
        META_REGION_OFFSET=meta_region_offset,
        NUM_SOA_FIELDS=NUM_SOA_FIELDS,
        SOA_K_NORM=0,
        SOA_V_SCALE=SOA_V_SCALE,
        SOA_V_ZERO=SOA_V_ZERO,
        BLOCK_D=HEAD_DIM,
        NORM_CORRECTION=0,
        FP8_E4B15=_use_fp8_e4b15(cache.device.index or 0),
        num_warps=4,
    )
    return key, value


def assemble_contract(
    query: torch.Tensor,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    cached_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = cached_len + query.shape[0]
    key = torch.empty(
        seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=query.dtype, device=query.device
    )
    value = torch.empty_like(key)
    key[:cached_len] = prefix_key[0, :, :cached_len, :].transpose(0, 1).to(query.dtype)
    value[:cached_len] = (
        prefix_value[0, :, :cached_len, :].transpose(0, 1).to(query.dtype)
    )
    key[cached_len:] = key_chunk
    value[cached_len:] = value_chunk
    return key, value


def sdpa_math(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cached_len: int,
    scale: float,
) -> torch.Tensor:
    q_len = query.shape[0]
    q_pos = torch.arange(q_len, device=query.device)[:, None] + cached_len
    k_pos = torch.arange(key.shape[0], device=query.device)[None, :]
    mask = k_pos <= q_pos
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
            scale=scale,
            enable_gqa=True,
        )[0].transpose(0, 1)


def online_replay(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cached_len: int,
    scale: float,
    *,
    qk_precision: str,
    pv_precision: str,
) -> torch.Tensor:
    """Replay the candidate tile order with controlled math precision."""
    q_len = query.shape[0]
    group = NUM_QUERY_HEADS // NUM_KV_HEADS
    q_grouped = query.reshape(q_len, NUM_KV_HEADS, group, HEAD_DIM).permute(1, 0, 2, 3)
    k_grouped = key.permute(1, 0, 2)
    v_grouped = value.permute(1, 0, 2)
    if qk_precision == "fp64":
        accum_dtype = torch.float64
        q_math = q_grouped.to(accum_dtype)
        k_math = k_grouped.to(accum_dtype)
    else:
        accum_dtype = torch.float32
        q_bf16 = q_grouped.to(torch.bfloat16)
        k_bf16 = k_grouped.to(torch.bfloat16)
        # Triton's BF16 dot accumulates the dot result in FP32 on this target.
        q_math = q_bf16.to(accum_dtype)
        k_math = k_bf16.to(accum_dtype)
    v_math = v_grouped.to(torch.float64 if pv_precision == "fp64" else torch.float32)
    e_max = torch.full(
        (NUM_KV_HEADS, q_len, group),
        -float("inf"),
        dtype=accum_dtype,
        device=query.device,
    )
    e_sum = torch.zeros_like(e_max)
    acc_dtype = torch.float64 if pv_precision == "fp64" else torch.float32
    acc = torch.zeros(
        NUM_KV_HEADS,
        q_len,
        group,
        HEAD_DIM,
        dtype=acc_dtype,
        device=query.device,
    )
    rcp_ln2 = 1.4426950408889634
    qk_scale = scale * rcp_ln2
    for start in range(0, key.shape[0], TILE_SIZE):
        end = min(start + TILE_SIZE, key.shape[0])
        scores = (
            torch.matmul(q_math, k_math[:, start:end].transpose(-1, -2)[:, None])
            * qk_scale
        )
        q_abs = cached_len + torch.arange(q_len, device=query.device)[:, None]
        k_abs = torch.arange(start, end, device=query.device)[None, :]
        scores = scores.masked_fill((k_abs > q_abs)[None, :, None, :], -float("inf"))
        next_max = torch.maximum(e_max, scores.amax(dim=-1))
        next_max = torch.where(
            next_max > -float("inf"), next_max, torch.zeros_like(next_max)
        )
        p = torch.exp2(scores - next_max[..., None])
        alpha = torch.exp2(e_max - next_max)
        e_sum = e_sum * alpha + p.sum(dim=-1)
        acc = acc * alpha[..., None]
        if pv_precision == "fp64":
            p_math = p.to(torch.float64)
        elif pv_precision == "fp32":
            p_math = p.to(torch.float32)
        else:
            p_math = p.to(torch.bfloat16).to(torch.float32)
        acc = acc + torch.matmul(p_math, v_math[:, start:end][:, None])
        e_max = next_max
    return (
        (acc / e_sum.clamp_min(1e-30)[..., None])
        .permute(1, 0, 2, 3)
        .reshape(q_len, NUM_QUERY_HEADS, HEAD_DIM)
        .to(query.dtype)
    )


def fp64_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cached_len: int,
    scale: float,
) -> torch.Tensor:
    """Compute a tiled FP64 reference without materializing all scores."""
    q_len = query.shape[0]
    q = (
        query.to(torch.float64)
        .reshape(q_len, NUM_KV_HEADS, -1, HEAD_DIM)
        .permute(1, 0, 2, 3)
    )
    k = key.to(torch.float64).permute(1, 0, 2)
    v = value.to(torch.float64).permute(1, 0, 2)
    q_pos = cached_len + torch.arange(q_len, device=query.device)[:, None]
    row_max = torch.full(
        (NUM_KV_HEADS, q_len, q.shape[2]),
        -float("inf"),
        dtype=torch.float64,
        device=query.device,
    )
    for start in range(0, key.shape[0], TILE_SIZE):
        end = min(start + TILE_SIZE, key.shape[0])
        scores = torch.matmul(q, k[:, start:end].transpose(-1, -2)[:, None]) * scale
        k_pos = torch.arange(start, end, device=query.device)[None, :]
        scores = scores.masked_fill((k_pos > q_pos)[None, :, None, :], -float("inf"))
        row_max = torch.maximum(row_max, scores.amax(dim=-1))
    row_sum = torch.zeros_like(row_max)
    for start in range(0, key.shape[0], TILE_SIZE):
        end = min(start + TILE_SIZE, key.shape[0])
        scores = torch.matmul(q, k[:, start:end].transpose(-1, -2)[:, None]) * scale
        k_pos = torch.arange(start, end, device=query.device)[None, :]
        scores = scores.masked_fill((k_pos > q_pos)[None, :, None, :], -float("inf"))
        row_sum += torch.exp(scores - row_max[..., None]).sum(dim=-1)
    output = torch.zeros(
        NUM_KV_HEADS,
        q_len,
        q.shape[2],
        HEAD_DIM,
        dtype=torch.float64,
        device=query.device,
    )
    for start in range(0, key.shape[0], TILE_SIZE):
        end = min(start + TILE_SIZE, key.shape[0])
        scores = torch.matmul(q, k[:, start:end].transpose(-1, -2)[:, None]) * scale
        k_pos = torch.arange(start, end, device=query.device)[None, :]
        scores = scores.masked_fill((k_pos > q_pos)[None, :, None, :], -float("inf"))
        probs = torch.exp(scores - row_max[..., None])
        output += torch.matmul(probs, v[:, start:end][:, None])
    output /= row_sum.clamp_min(1e-300)[..., None]
    return (
        output.permute(1, 0, 2, 3)
        .reshape(q_len, NUM_QUERY_HEADS, HEAD_DIM)
        .to(query.dtype)
    )


@triton.jit
def _dump_candidate_cache_tiles(
    cache_ptr,
    cache_u16_ptr,
    block_table_ptr,
    out_k_ptr,
    out_v_ptr,
    stride_cache_block: tl.int64,
    stride_bt: tl.int64,
    stride_out_token: tl.int64,
    stride_out_head: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    DATA_BYTES_PER_SLOT: tl.constexpr,
    META_REGION_OFFSET: tl.constexpr,
    NUM_SOA_FIELDS: tl.constexpr,
    SOA_V_SCALE: tl.constexpr,
    SOA_V_ZERO: tl.constexpr,
    FP8_E4B15: tl.constexpr,
):
    token = tl.program_id(0)
    kv_hid = tl.program_id(1)
    block = token // BLOCK_SIZE
    slot = token % BLOCK_SIZE
    physical = tl.load(block_table_ptr + block * stride_bt).to(tl.int64)
    block_base = physical * stride_cache_block
    data_base = (
        block_base
        + slot * (NUM_KV_HEADS * DATA_BYTES_PER_SLOT)
        + kv_hid * DATA_BYTES_PER_SLOT
    )
    d = tl.arange(0, HEAD_DIM)
    raw_k = tl.load(cache_ptr + data_base + d)
    if FP8_E4B15:
        k = raw_k.to(tl.float8e4b15, bitcast=True).to(tl.float32)
    else:
        k = raw_k.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    half_d = tl.arange(0, HEAD_DIM // 2)
    raw_v = tl.load(cache_ptr + data_base + KEY_DATA_BYTES + half_d).to(tl.int32)
    lo = (raw_v & 0xF).to(tl.float32)
    hi = ((raw_v >> 4) & 0xF).to(tl.float32)
    value_indices = tl.interleave(lo, hi)
    meta_base = (block_base + META_REGION_OFFSET) // 2 + kv_hid * (
        NUM_SOA_FIELDS * BLOCK_SIZE
    )
    scale_addr = meta_base + SOA_V_SCALE * BLOCK_SIZE + slot
    zero_addr = meta_base + SOA_V_ZERO * BLOCK_SIZE + slot
    scale_bits = tl.load(cache_u16_ptr + scale_addr)
    zero_bits = tl.load(cache_u16_ptr + zero_addr)
    scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
    zero = zero_bits.to(tl.float16, bitcast=True).to(tl.float32)
    v = value_indices * scale + zero
    out_base = token * stride_out_token + kv_hid * stride_out_head
    tl.store(out_k_ptr + out_base + d, k)
    tl.store(out_v_ptr + out_base + d, v)


def dump_candidate_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cached_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = int(cache.shape[1])
    out_k = torch.empty(
        cached_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32, device=cache.device
    )
    out_v = torch.empty_like(out_k)
    meta_region_offset = block_size * NUM_KV_HEADS * DATA_BYTES_PER_SLOT
    _dump_candidate_cache_tiles[(cached_len, NUM_KV_HEADS)](
        cache,
        cache.view(torch.uint16),
        block_table,
        out_k,
        out_v,
        cache.stride(0),
        block_table.stride(1),
        out_k.stride(0),
        out_k.stride(1),
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=NUM_KV_HEADS,
        HEAD_DIM=HEAD_DIM,
        KEY_DATA_BYTES=KEY_DATA_BYTES,
        DATA_BYTES_PER_SLOT=DATA_BYTES_PER_SLOT,
        META_REGION_OFFSET=meta_region_offset,
        NUM_SOA_FIELDS=NUM_SOA_FIELDS,
        SOA_V_SCALE=SOA_V_SCALE,
        SOA_V_ZERO=SOA_V_ZERO,
        FP8_E4B15=_use_fp8_e4b15(cache.device.index or 0),
        num_warps=4,
    )
    return out_k, out_v


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-outputs", type=Path)
    args = parser.parse_args()
    saved = torch.load(args.input, map_location="cpu", weights_only=False)
    query = saved["query"].cuda()
    key_chunk = saved["key_chunk"].cuda()
    value_chunk = saved["value_chunk"].cuda()
    cache = saved["kv_cache"].cuda()
    block_table = saved["block_table"].cuda()
    cached_len = int(saved["cached_len"])
    seq_len = int(saved["seq_len"])
    scale = float(saved["scale"])
    prefix_key, prefix_value = dequant_prefix(cache, block_table, cached_len)
    key, value = assemble_contract(
        query, key_chunk, value_chunk, prefix_key, prefix_value, cached_len
    )
    old = sdpa_math(query, key, value, cached_len, scale)
    candidate = launch_gfx1201_tq_continuation_prefill(
        query=query,
        key_chunk=key_chunk,
        value_chunk=value_chunk,
        kv_cache=cache,
        block_table=block_table,
        cached_len=cached_len,
        seq_len=seq_len,
        scale=scale,
    )
    variants = {
        "old_sdpa_math": old,
        "candidate_streaming": candidate,
        "online_pv_fp32": online_replay(
            query,
            key,
            value,
            cached_len,
            scale,
            qk_precision="bf16",
            pv_precision="fp32",
        ),
        "online_pv_fp64": online_replay(
            query,
            key,
            value,
            cached_len,
            scale,
            qk_precision="bf16",
            pv_precision="fp64",
        ),
        "online_qk_pv_fp64": online_replay(
            query,
            key,
            value,
            cached_len,
            scale,
            qk_precision="fp64",
            pv_precision="fp64",
        ),
        "fp64_reference": fp64_reference(query, key, value, cached_len, scale),
    }
    decoded_key, decoded_value = dump_candidate_cache(cache, block_table, cached_len)
    prefix_key_contract = prefix_key[0, :, :cached_len, :].transpose(0, 1)
    prefix_value_contract = prefix_value[0, :, :cached_len, :].transpose(0, 1)
    decode_results = {
        "key_fp16": metrics(decoded_key.to(torch.float16), prefix_key_contract),
        "value_fp16": metrics(decoded_value.to(torch.float16), prefix_value_contract),
        "key_query_dtype": metrics(
            decoded_key.to(torch.float16).to(query.dtype),
            prefix_key_contract.to(query.dtype),
        ),
        "value_query_dtype": metrics(
            decoded_value.to(torch.float16).to(query.dtype),
            prefix_value_contract.to(query.dtype),
        ),
    }
    result = {
        "input": str(args.input),
        "layer": saved.get("layer"),
        "query_shape": list(query.shape),
        "cache_shape": list(cache.shape),
        "cached_len": cached_len,
        "seq_len": seq_len,
        "dtype": str(query.dtype),
        "block_size": int(cache.shape[1]),
        "dequant": {
            "method": "candidate_kernel_load_replay",
            "fp8_e4b15": bool(_use_fp8_e4b15(cache.device.index or 0)),
        },
        "captured_baseline_vs_old": metrics(saved["baseline_output"].cuda(), old),
        "variants_vs_old": {
            name: metrics(output, old) for name, output in variants.items()
        },
        "variants_vs_fp64": {
            name: metrics(output, variants["fp64_reference"])
            for name, output in variants.items()
        },
        "cache_decode_vs_full_dequant": decode_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.save_outputs:
        args.save_outputs.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {name: output.detach().cpu() for name, output in variants.items()},
            args.save_outputs,
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
