# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Specialized gfx1201 TurboQuant K8/V4 decode kernels.

The first implementation is deliberately narrow: it handles one query token
per request for the exact D=256, Hq/Hk=6 profile. It reads the existing SoA
cache layout and writes the same [B, Hq, splits, D + 1] partial format as the
generic TurboQuant decoder, which lets it reuse the established stage-2
online-softmax reducer.

The query-start-location pointer is part of the stage-1 contract even in this
single-token phase. The kernel uses it to locate each request's query row, so
the later multi-token extension will not need a new launcher ABI.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

from .triton_turboquant_decode import _use_fp8_e4b15

# Exact specialization constants. Keep these compile-time in the first kernel.
TARGET_HEAD_SIZE = 256
TARGET_GQA_GROUP_SIZE = 6
TARGET_VALUE_QUANT_BITS = 4

# Existing TurboQuant SoA K8/V4 D=256 layout.
KEY_DATA_BYTES = 256
VALUE_DATA_BYTES = 128
DATA_BYTES_PER_SLOT = KEY_DATA_BYTES + VALUE_DATA_BYTES  # 384
NUM_SOA_FIELDS = 2  # V_SCALE, V_ZERO
SOA_V_SCALE = 0
SOA_V_ZERO = 1
LOGICAL_BYTES_PER_SLOT = DATA_BYTES_PER_SLOT + NUM_SOA_FIELDS * 2  # 388

BLOCK_M = 16
TILE_SIZE = 16


@triton.jit
def _gfx1201_k8v4_stage1(
    Query_ptr,
    KV_cache_ptr,
    KV_cache_u16_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Query_start_loc_ptr,
    Mid_o_ptr,
    stride_qb: tl.int64,
    stride_qh: tl.int64,
    stride_cache_block: tl.int64,
    stride_bt_b: tl.int64,
    stride_mid_b: tl.int64,
    stride_mid_h: tl.int64,
    stride_mid_s: tl.int64,
    scale,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
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
):
    """Fused K8/V4 stage 1 for one token per request.

    One program owns a KV head and all six query heads in its GQA group. The
    16-wide row tile is intentional: the six valid rows provide the grouped-Q
    work, while the padded rows keep the dot shapes friendly to Triton MFMA
    lowering on gfx1201.
    """
    bid = tl.program_id(0)
    kv_hid = tl.program_id(1)
    split_id = tl.program_id(2)

    seq_len = tl.load(Seq_lens_ptr + bid).to(tl.int64)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * split_id
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    row_offs = tl.arange(0, BLOCK_M)
    row_mask = row_offs < KV_GROUP_SIZE
    kv_offs_in_tile = tl.arange(0, TILE_SIZE)

    # In the single-token phase query_start_loc[b] is the query row for b.
    # Reading it here preserves the direct query-start-location contract and
    # avoids constructing a synthetic arange in the Python launcher.
    query_token = tl.load(Query_start_loc_ptr + bid).to(tl.int64)
    q_heads = (kv_hid * KV_GROUP_SIZE + row_offs).to(tl.int64)
    q_addrs = query_token * stride_qb + q_heads[:, None] * stride_qh + d_offs[None, :]
    q = tl.load(
        Query_ptr + q_addrs,
        mask=row_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    q_dot = q.to(tl.bfloat16) if USE_BF16_DOT else q.to(tl.float16)

    # Online softmax is kept in log2 space. The stage-2 reducer expects a
    # natural-log LSE, so the epilogue converts the maximum back with LN2.
    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453
    qk_scale = scale * RCP_LN2
    e_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    block_table_base = bid * stride_bt_b
    for start_n in range(split_start, split_end, TILE_SIZE):
        kv_offs = start_n + kv_offs_in_tile
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        physical_block = tl.load(
            Block_table_ptr + block_table_base + page_idx,
            mask=kv_mask,
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

        # K8: one FP8 byte per dimension, no rotation or centroid lookup.
        k_raw = tl.load(
            KV_cache_ptr + data_bases[:, None] + d_offs[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        )
        if FP8_E4B15:
            k = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        k_t = tl.trans(k)
        if USE_BF16_DOT:
            scores = qk_scale * tl.dot(q_dot, k_t.to(tl.bfloat16))
        else:
            scores = qk_scale * tl.dot(q_dot, k_t.to(tl.float16))
        scores = tl.where(row_mask[:, None] & kv_mask[None, :], scores, float("-inf"))

        next_max = tl.maximum(e_max, tl.max(scores, axis=1))
        next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
        p = tl.math.exp2(scores - next_max[:, None])
        alpha = tl.math.exp2(e_max - next_max)
        e_sum = e_sum * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # V4: each packed byte is loaded once, then its two nibbles are
        # interleaved back into the D-dimensional value vector.
        half_d = tl.arange(0, BLOCK_D // 2)
        half_mask = half_d * 2 < HEAD_DIM
        value_bytes = tl.load(
            KV_cache_ptr + data_bases[:, None] + KEY_DATA_BYTES + half_d[None, :],
            mask=kv_mask[:, None] & half_mask[None, :],
            other=0,
        ).to(tl.int32)
        value_lo = (value_bytes & 0xF).to(tl.float32)
        value_hi = ((value_bytes >> 4) & 0xF).to(tl.float32)
        value_indices = tl.interleave(value_lo, value_hi)

        v_scale_u16 = tl.load(KV_cache_u16_ptr + scale_addrs, mask=kv_mask, other=0)
        v_zero_u16 = tl.load(KV_cache_u16_ptr + zero_addrs, mask=kv_mask, other=0)
        v_scale = v_scale_u16.to(tl.float16, bitcast=True).to(tl.float32)
        v_zero = v_zero_u16.to(tl.float16, bitcast=True).to(tl.float32)
        v = value_indices * v_scale[:, None] + v_zero[:, None]

        if USE_BF16_DOT:
            acc += tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        else:
            acc += tl.dot(p.to(tl.float16), v.to(tl.float16))
        e_max = next_max

    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    acc = acc / safe_sum[:, None]
    lse = e_max * LN2 + tl.log(safe_sum)

    out_addrs = (
        bid * stride_mid_b
        + q_heads[:, None] * stride_mid_h
        + split_id * stride_mid_s
        + d_offs[None, :]
    )
    tl.store(
        Mid_o_ptr + out_addrs,
        acc,
        mask=row_mask[:, None] & d_mask[None, :],
    )
    lse_addrs = (
        bid * stride_mid_b + q_heads * stride_mid_h + split_id * stride_mid_s + HEAD_DIM
    )
    tl.store(Mid_o_ptr + lse_addrs, lse, mask=row_mask)


def _validate_single_token_inputs(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_num_kv_splits: int,
) -> tuple[int, int, int]:
    """Validate the fixed specialization before launching any Triton code."""
    if query.ndim != 3:
        raise ValueError(f"query must have shape [B, Hq, D], got {query.shape}")
    batch, num_query_heads, head_size = query.shape
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"gfx1201 K8/V4 decode requires fp16 or bf16 queries, got {query.dtype}"
        )
    if head_size != TARGET_HEAD_SIZE:
        raise ValueError(
            f"gfx1201 K8/V4 decode requires D={TARGET_HEAD_SIZE}, got {head_size}"
        )
    if kv_cache.ndim != 4 or kv_cache.dtype != torch.uint8:
        raise ValueError("kv_cache must be a rank-4 uint8 tensor")
    if kv_cache.shape[2] * TARGET_GQA_GROUP_SIZE != num_query_heads:
        raise ValueError(
            "gfx1201 K8/V4 decode requires Hq/Hk=6, got "
            f"Hq={num_query_heads}, Hk={kv_cache.shape[2]}"
        )
    if kv_cache.shape[-1] < LOGICAL_BYTES_PER_SLOT:
        raise ValueError(
            f"K8/V4 SoA cache needs at least {LOGICAL_BYTES_PER_SLOT} bytes "
            f"per slot, got {kv_cache.shape[-1]}"
        )
    block_size = int(kv_cache.shape[1])
    if block_size not in (16, 32):
        raise ValueError(
            f"gfx1201 K8/V4 decode supports block sizes 16 and 32, got {block_size}"
        )
    if block_table.ndim != 2 or block_table.shape[0] < batch:
        raise ValueError("block_table must have shape [B, max_num_blocks]")
    if seq_lens.ndim != 1 or seq_lens.shape[0] < batch:
        raise ValueError("seq_lens must have shape [B]")
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] != batch + 1:
        raise ValueError(
            "query_start_loc must have shape [B + 1] for the decode contract"
        )
    devices = {
        query.device,
        kv_cache.device,
        block_table.device,
        seq_lens.device,
        query_start_loc.device,
    }
    if len(devices) != 1:
        raise ValueError("query, cache, and attention metadata must share a device")
    if max_num_kv_splits < 1:
        raise ValueError("max_num_kv_splits must be positive")

    # The first phase is intentionally single-token only. The backend keeps
    # this path behind supports_spec_as_decode=False, but validate CPU metadata
    # here as well so direct callers fail closed instead of silently using the
    # first query token of an MTP request.
    if query_start_loc.device.type == "cpu":
        qsl = query_start_loc.tolist()
        if qsl != list(range(batch + 1)):
            raise ValueError(
                "gfx1201 single-token decode requires query_start_loc="
                f"[0, 1, ..., B], got {qsl}"
            )
    return batch, num_query_heads, block_size


def triton_turboquant_decode_gfx1201_k8v4(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    scale: float,
    *,
    output: torch.Tensor | None = None,
    mid_o_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    max_num_kv_splits: int = 8,
    max_seq_len: int = 0,
) -> torch.Tensor:
    """Run the specialized single-token gfx1201 K8/V4 decode.

    Args:
        query: Query tensor with shape [B, Hq, 256].
        kv_cache: Existing TurboQuant SoA cache with shape
            [num_blocks, block_size, Hk, slot_size].
        block_table: Physical block table with shape [B, max_blocks].
        seq_lens: Number of visible KV entries for each request.
        query_start_loc: Query cumulative starts. In this phase it must be
            [0, 1, ..., B] and is passed directly to stage 1.
        scale: Attention scale, normally 1 / sqrt(256).
        output: Optional final output buffer with shape [B, Hq, 256].
        mid_o_buf: Optional reusable fp32 stage-1 buffer with shape at least
            [B, Hq, max_num_kv_splits, 257].
        lse_buf: Optional reusable fp32 LSE buffer with shape at least [B, Hq].
        max_num_kv_splits: Fixed compile-time split count for this launcher.
        max_seq_len: Retained for backend API parity; sequence lengths are
            read from seq_lens by the kernel.

    Returns:
        Attention output in the query dtype.

    Raises:
        ValueError: If the tensors do not satisfy the fixed target profile.
    """
    del max_seq_len
    batch, num_query_heads, block_size = _validate_single_token_inputs(
        query,
        kv_cache,
        block_table,
        seq_lens,
        query_start_loc,
        max_num_kv_splits,
    )
    _, _, head_size = query.shape
    num_kv_heads = kv_cache.shape[2]
    device = query.device
    num_splits = int(max_num_kv_splits)

    bytes_per_block = block_size * num_kv_heads * LOGICAL_BYTES_PER_SLOT
    if kv_cache.stride(0) < bytes_per_block:
        raise ValueError(
            "kv_cache stride(0) is smaller than the target SoA block layout: "
            f"stride={kv_cache.stride(0)}, required={bytes_per_block}"
        )

    if mid_o_buf is not None:
        if (
            mid_o_buf.ndim != 4
            or mid_o_buf.dtype != torch.float32
            or mid_o_buf.device != device
            or mid_o_buf.shape[0] < batch
            or mid_o_buf.shape[1] < num_query_heads
            or mid_o_buf.shape[2] < num_splits
            or mid_o_buf.shape[3] < head_size + 1
        ):
            raise ValueError(
                "mid_o_buf must be an fp32 [B, Hq, splits, D + 1] workspace"
            )
        mid_o = mid_o_buf[:batch, :num_query_heads, :num_splits, : head_size + 1]
    else:
        mid_o = torch.empty(
            batch,
            num_query_heads,
            num_splits,
            head_size + 1,
            dtype=torch.float32,
            device=device,
        )

    if output is None:
        output = torch.empty(
            batch,
            num_query_heads,
            head_size,
            dtype=query.dtype,
            device=device,
        )
    elif (
        output.ndim != 3
        or output.device != device
        or output.shape[0] < batch
        or output.shape[1] < num_query_heads
        or output.shape[2] < head_size
    ):
        raise ValueError("output must be a [B, Hq, D] buffer on query.device")
    output_view = output[:batch, :num_query_heads, :head_size]

    if lse_buf is not None:
        if (
            lse_buf.ndim != 2
            or lse_buf.dtype != torch.float32
            or lse_buf.device != device
            or lse_buf.shape[0] < batch
            or lse_buf.shape[1] < num_query_heads
        ):
            raise ValueError("lse_buf must be an fp32 [B, Hq] workspace")
        lse = lse_buf[:batch, :num_query_heads]
    else:
        lse = torch.empty(batch, num_query_heads, dtype=torch.float32, device=device)

    kv_cache_u16 = kv_cache.view(torch.uint16)
    _launch_single_token_stage1(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        scale=scale,
        mid_o_buf=mid_o,
        num_kv_splits=num_splits,
        kv_cache_u16=kv_cache_u16,
    )

    _fwd_kernel_stage2[(batch, num_query_heads)](
        mid_o,
        output_view,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output_view.stride(0),
        output_view.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=num_splits,
        BLOCK_DV=triton.next_power_of_2(head_size),
        Lv=head_size,
        num_warps=4,
        num_stages=2,
    )
    return output_view


def _launch_single_token_stage1(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    scale: float,
    mid_o_buf: torch.Tensor,
    *,
    num_kv_splits: int,
    kv_cache_u16: torch.Tensor | None = None,
) -> None:
    """Launch the fixed-shape grouped-Q stage-1 kernel."""
    num_kv_heads = kv_cache.shape[2]
    block_size = int(kv_cache.shape[1])
    meta_region_offset = block_size * num_kv_heads * DATA_BYTES_PER_SLOT
    if kv_cache_u16 is None:
        kv_cache_u16 = kv_cache.view(torch.uint16)

    _gfx1201_k8v4_stage1[(query.shape[0], num_kv_heads, num_kv_splits)](
        query,
        kv_cache,
        kv_cache_u16,
        block_table,
        seq_lens,
        query_start_loc,
        mid_o_buf,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        block_table.stride(0),
        mid_o_buf.stride(0),
        mid_o_buf.stride(1),
        mid_o_buf.stride(2),
        scale,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=num_kv_heads,
        NUM_KV_SPLITS=num_kv_splits,
        KV_GROUP_SIZE=TARGET_GQA_GROUP_SIZE,
        HEAD_DIM=TARGET_HEAD_SIZE,
        BLOCK_D=TARGET_HEAD_SIZE,
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
        num_warps=4,
        num_stages=1,
    )


def _launch_multi_token_stage1(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    scale: float,
    mid_o_buf: torch.Tensor,
    *,
    num_kv_splits: int,
) -> None:
    """Reserve the Phase 5 entry point without enabling MTP prematurely."""
    del query, kv_cache, block_table, seq_lens, query_start_loc, scale
    del mid_o_buf, num_kv_splits
    raise NotImplementedError(
        "gfx1201 K8/V4 multi-token decode is not implemented yet; "
        "keep supports_spec_as_decode disabled"
    )
