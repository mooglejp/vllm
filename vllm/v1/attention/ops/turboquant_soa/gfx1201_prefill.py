# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw-current gfx1201 TurboQuant continuation-prefill candidate.

The kernel intentionally keeps the existing gfx1201 K8/V4 stage-1 shape:
GQA query tiles share a K/V tile and one online-softmax state.  Prefix
positions are decoded directly from the SoA cache while current-chunk
positions are loaded from raw K/V tensors.  The launcher is not imported by
the production backend unless its explicit opt-in gate is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from vllm.triton_utils import tl, triton

from .triton_turboquant_decode import _use_fp8_e4b15
from .triton_turboquant_decode_gfx1201_k8v4 import (
    DATA_BYTES_PER_SLOT,
    KEY_DATA_BYTES,
    LOGICAL_BYTES_PER_SLOT,
    NUM_SOA_FIELDS,
    SOA_V_SCALE,
    SOA_V_ZERO,
    TARGET_GQA_GROUP_SIZE,
)

HEAD_DIM: Final = 256
GQA: Final = TARGET_GQA_GROUP_SIZE
BLOCK_M: Final = 32
TILE_SIZE: Final = 16


@dataclass(frozen=True, slots=True)
class Gfx1201TurboQuantPrefillContract:
    """Narrow raw-current continuation contract."""

    head_dim: int = HEAD_DIM
    gqa: int = GQA
    causal: bool = True
    supports_sinks: bool = False
    supports_sliding_window: bool = False
    max_num_kv_splits: int = 1


@triton.jit
def _gfx1201_k8v4_raw_current_stage1(
    Query_ptr,
    Key_chunk_ptr,
    Value_chunk_ptr,
    KV_cache_ptr,
    KV_cache_u16_ptr,
    Block_table_ptr,
    Output_ptr,
    stride_qt: tl.int64,
    stride_qh: tl.int64,
    stride_kt: tl.int64,
    stride_kh: tl.int64,
    stride_vt: tl.int64,
    stride_vh: tl.int64,
    stride_cache_block: tl.int64,
    stride_bt: tl.int64,
    stride_out_token: tl.int64,
    stride_oh: tl.int64,
    cached_len,
    query_len,
    seq_len,
    scale,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    DATA_BYTES_PER_SLOT: tl.constexpr,
    META_REGION_OFFSET: tl.constexpr,
    NUM_SOA_FIELDS: tl.constexpr,
    SOA_V_SCALE: tl.constexpr,
    SOA_V_ZERO: tl.constexpr,
    FP8_E4B15: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    QUERY_BLOCK_SIZE: tl.constexpr,
):
    """One splitless online-softmax program for a query tile and KV head."""
    query_block_id = tl.program_id(0)
    kv_hid = tl.program_id(1)
    query_block_start = query_block_id * QUERY_BLOCK_SIZE
    if query_block_start >= query_len:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    row_offs = tl.arange(0, BLOCK_M)
    row_mask = row_offs < QUERY_BLOCK_SIZE * KV_GROUP_SIZE
    kv_offs_in_tile = tl.arange(0, TILE_SIZE)

    query_pos = query_block_start + row_offs // KV_GROUP_SIZE
    query_mask = row_mask & (query_pos < query_len)
    q_heads = (kv_hid * KV_GROUP_SIZE + row_offs % KV_GROUP_SIZE).to(tl.int64)
    q_addrs = (
        query_pos[:, None] * stride_qt + q_heads[:, None] * stride_qh + d_offs[None, :]
    )
    q = tl.load(
        Query_ptr + q_addrs,
        mask=query_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    q_dot = q.to(tl.bfloat16) if USE_BF16_DOT else q.to(tl.float16)
    query_abs_pos = cached_len + query_pos

    RCP_LN2: tl.constexpr = 1.4426950408889634
    qk_scale = scale * RCP_LN2
    e_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, seq_len, TILE_SIZE):
        kv_offs = start_n + kv_offs_in_tile
        kv_mask = kv_offs < seq_len
        cache_mask = kv_mask & (kv_offs < cached_len)
        raw_idx = kv_offs - cached_len
        raw_idx_safe = tl.maximum(raw_idx, 0)
        raw_mask = kv_mask & (raw_idx >= 0) & (raw_idx < query_len)

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        physical_block = tl.load(
            Block_table_ptr + page_idx * stride_bt,
            mask=cache_mask,
            other=0,
        ).to(tl.int64)
        block_base = physical_block * stride_cache_block
        slot_off = page_off.to(tl.int64)
        data_bases = (
            block_base
            + slot_off * (NUM_KV_HEADS * DATA_BYTES_PER_SLOT)
            + tl.cast(kv_hid, tl.int64) * DATA_BYTES_PER_SLOT
        )

        meta_base = (block_base + META_REGION_OFFSET) // 2 + tl.cast(
            kv_hid, tl.int64
        ) * (NUM_SOA_FIELDS * BLOCK_SIZE)
        scale_addrs = meta_base + SOA_V_SCALE * BLOCK_SIZE + slot_off
        zero_addrs = meta_base + SOA_V_ZERO * BLOCK_SIZE + slot_off

        k_cache_raw = tl.load(
            KV_cache_ptr + data_bases[:, None] + d_offs[None, :],
            mask=cache_mask[:, None] & d_mask[None, :],
            other=0,
        )
        if FP8_E4B15:
            k_cache = k_cache_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k_cache = k_cache_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        k_current = tl.load(
            Key_chunk_ptr
            + raw_idx_safe[:, None] * stride_kt
            + kv_hid * stride_kh
            + d_offs[None, :],
            mask=raw_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        # Match the current continuation contract: cached values are rounded
        # through FP16 before conversion to the query dtype; raw values are not.
        if USE_BF16_DOT:
            k = tl.where(
                cache_mask[:, None],
                k_cache.to(tl.float16).to(tl.bfloat16),
                k_current.to(tl.bfloat16),
            )
        else:
            k = tl.where(cache_mask[:, None], k_cache.to(tl.float16), k_current)
        k_t = tl.trans(k)
        if USE_BF16_DOT:
            scores = qk_scale * tl.dot(q_dot, k_t.to(tl.bfloat16))
        else:
            scores = qk_scale * tl.dot(q_dot, k_t.to(tl.float16))
        scores = tl.where(
            query_mask[:, None]
            & kv_mask[None, :]
            & (kv_offs[None, :] <= query_abs_pos[:, None]),
            scores,
            float("-inf"),
        )

        next_max = tl.maximum(e_max, tl.max(scores, axis=1))
        next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
        p = tl.math.exp2(scores - next_max[:, None])
        alpha = tl.math.exp2(e_max - next_max)
        e_sum = e_sum * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        half_d = tl.arange(0, BLOCK_D // 2)
        half_mask = half_d * 2 < HEAD_DIM
        value_bytes = tl.load(
            KV_cache_ptr + data_bases[:, None] + KEY_DATA_BYTES + half_d[None, :],
            mask=cache_mask[:, None] & half_mask[None, :],
            other=0,
        ).to(tl.int32)
        value_lo = (value_bytes & 0xF).to(tl.float32)
        value_hi = ((value_bytes >> 4) & 0xF).to(tl.float32)
        value_indices = tl.interleave(value_lo, value_hi)
        v_scale_u16 = tl.load(
            KV_cache_u16_ptr + scale_addrs,
            mask=cache_mask,
            other=0,
        )
        v_zero_u16 = tl.load(
            KV_cache_u16_ptr + zero_addrs,
            mask=cache_mask,
            other=0,
        )
        v_cache = (
            value_indices
            * v_scale_u16.to(tl.float16, bitcast=True).to(tl.float32)[:, None]
            + v_zero_u16.to(tl.float16, bitcast=True).to(tl.float32)[:, None]
        )
        v_current = tl.load(
            Value_chunk_ptr
            + raw_idx_safe[:, None] * stride_vt
            + kv_hid * stride_vh
            + d_offs[None, :],
            mask=raw_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if USE_BF16_DOT:
            v = tl.where(
                cache_mask[:, None],
                v_cache.to(tl.float16).to(tl.bfloat16),
                v_current.to(tl.bfloat16),
            )
            acc += tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        else:
            v = tl.where(cache_mask[:, None], v_cache.to(tl.float16), v_current)
            acc += tl.dot(p.to(tl.float16), v.to(tl.float16))
        e_max = next_max

    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    acc = acc / safe_sum[:, None]
    out_addrs = (
        query_pos[:, None] * stride_out_token
        + q_heads[:, None] * stride_oh
        + d_offs[None, :]
    )
    tl.store(
        Output_ptr + out_addrs,
        acc,
        mask=query_mask[:, None] & d_mask[None, :],
    )


def is_gfx1201_tq_prefill_candidate(
    *,
    query: torch.Tensor,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    cached_len: int,
) -> bool:
    """Return whether tensors match the opt-in P2.2 raw-current profile."""
    if query.ndim != 3 or key_chunk.ndim != 3 or value_chunk.ndim != 3:
        return False
    q_len, num_query_heads, head_dim = query.shape
    return (
        cached_len > 0
        and q_len > 128
        and head_dim == HEAD_DIM
        and key_chunk.shape[-1] == HEAD_DIM
        and value_chunk.shape[-1] == HEAD_DIM
        and num_query_heads == key_chunk.shape[1] * GQA
        and key_chunk.shape == value_chunk.shape
        and key_chunk.shape[0] == q_len
        and query.dtype in (torch.float16, torch.bfloat16)
        and key_chunk.dtype == query.dtype
        and value_chunk.dtype == query.dtype
        and query.stride(-1) == 1
        and key_chunk.stride(-1) == 1
        and value_chunk.stride(-1) == 1
    )


def launch_gfx1201_tq_continuation_prefill(
    *,
    query: torch.Tensor,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cached_len: int,
    seq_len: int,
    scale: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch splitless raw-current K8/V4 continuation attention.

    Prefix positions are read from the existing K8/V4 SoA cache and current
    positions are read from ``key_chunk``/``value_chunk``.  One online-softmax
    state covers both ranges, so no full-prefix K/V or split-local softmax is
    materialized.
    """
    if not is_gfx1201_tq_prefill_candidate(
        query=query,
        key_chunk=key_chunk,
        value_chunk=value_chunk,
        cached_len=cached_len,
    ):
        raise ValueError("tensors do not match the gfx1201 raw-current profile")
    q_len, num_query_heads, head_dim = query.shape
    if seq_len != cached_len + q_len:
        raise ValueError("seq_len must equal cached_len + query length")
    if kv_cache.ndim != 4 or kv_cache.dtype != torch.uint8:
        raise ValueError("kv_cache must be a rank-4 uint8 tensor")
    block_size = int(kv_cache.shape[1])
    if block_size not in (16, 32):
        raise ValueError("gfx1201 raw-current prefill supports block sizes 16 and 32")
    num_kv_heads = int(key_chunk.shape[1])
    if kv_cache.shape[2] != num_kv_heads:
        raise ValueError("cache and raw current chunk must have the same Hk")
    if kv_cache.shape[-1] < LOGICAL_BYTES_PER_SLOT:
        raise ValueError(
            f"K8/V4 SoA cache needs at least {LOGICAL_BYTES_PER_SLOT} bytes per slot"
        )
    if block_table.ndim != 2 or block_table.shape[0] < 1:
        raise ValueError("block_table must have shape [1, max_num_blocks]")
    needed_blocks = (seq_len + block_size - 1) // block_size
    if block_table.shape[1] < needed_blocks:
        raise ValueError("block_table does not cover the visible sequence")
    devices = {query.device, key_chunk.device, value_chunk.device, kv_cache.device}
    devices.add(block_table.device)
    if len(devices) != 1:
        raise ValueError("query, raw K/V, cache, and block table must share a device")
    bytes_per_block = block_size * num_kv_heads * LOGICAL_BYTES_PER_SLOT
    if kv_cache.stride(0) < bytes_per_block:
        raise ValueError("kv_cache stride(0) is smaller than the SoA block layout")

    if output is None:
        output = torch.empty_like(query)
    elif (
        output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
    ):
        raise ValueError("output must have the same shape, dtype, and device as query")
    if output.stride(-1) != 1:
        raise ValueError("output's last dimension must have unit stride")

    meta_region_offset = block_size * num_kv_heads * DATA_BYTES_PER_SLOT
    kv_cache_u16 = kv_cache.view(torch.uint16)
    query_block_size = 4
    num_query_blocks = (q_len + query_block_size - 1) // query_block_size
    _gfx1201_k8v4_raw_current_stage1[(num_query_blocks, num_kv_heads)](
        query,
        key_chunk,
        value_chunk,
        kv_cache,
        kv_cache_u16,
        block_table,
        output,
        query.stride(0),
        query.stride(1),
        key_chunk.stride(0),
        key_chunk.stride(1),
        value_chunk.stride(0),
        value_chunk.stride(1),
        kv_cache.stride(0),
        block_table.stride(1),
        output.stride(0),
        output.stride(1),
        cached_len,
        q_len,
        seq_len,
        scale,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=num_kv_heads,
        KV_GROUP_SIZE=GQA,
        HEAD_DIM=head_dim,
        BLOCK_D=head_dim,
        BLOCK_M=BLOCK_M,
        TILE_SIZE=TILE_SIZE,
        KEY_DATA_BYTES=KEY_DATA_BYTES,
        DATA_BYTES_PER_SLOT=DATA_BYTES_PER_SLOT,
        META_REGION_OFFSET=meta_region_offset,
        NUM_SOA_FIELDS=NUM_SOA_FIELDS,
        SOA_V_SCALE=SOA_V_SCALE,
        SOA_V_ZERO=SOA_V_ZERO,
        FP8_E4B15=_use_fp8_e4b15(query.device.index or 0),
        USE_BF16_DOT=1 if query.dtype == torch.bfloat16 else 0,
        QUERY_BLOCK_SIZE=query_block_size,
        num_warps=4,
        num_stages=2,
    )
    return output
