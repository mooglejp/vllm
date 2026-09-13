# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic-only P2.2 continuation capture sitecustomize.

This module is loaded only when its directory is prepended to PYTHONPATH for a
local model run. It disables flash prefill and captures one baseline
continuation call; it does not alter production source or dispatch settings.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

# Keep the capture run deterministic with the same diagnostic environment used
# by the existing greedy harness.
from vllm.model_executor.warmup import kernel_warmup
from vllm.v1.attention.backends import turboquant_attn
from vllm.v1.worker.gpu import warmup

warmup.warmup_kernels = lambda *args, **kwargs: None
kernel_warmup.kernel_warmup = lambda *args, **kwargs: None
turboquant_attn._HAS_FLASH_ATTN = False


_CAPTURED = False


def _compact_cache(cache: torch.Tensor, block_table: torch.Tensor, seq_len: int):
    block_size = int(cache.shape[1])
    num_blocks = math.ceil(seq_len / block_size)
    physical = block_table[0, :num_blocks].to(torch.long)
    slot_bytes = int(cache.shape[-1])
    raw_bytes = block_size * int(cache.shape[2]) * slot_bytes
    raw_blocks = cache.as_strided((cache.shape[0], raw_bytes), (cache.stride(0), 1))
    compact = raw_blocks[physical].reshape(
        num_blocks, block_size, int(cache.shape[2]), slot_bytes
    )
    return compact.contiguous(), torch.arange(
        num_blocks, dtype=torch.int32, device=cache.device
    )[None]


def install() -> None:
    global _CAPTURED
    destination = os.environ.get("TQ_P22_PREFILL_CAPTURE_OUTPUT")
    if not destination:
        return
    output_path = Path(destination)
    layer_filter = os.environ.get(
        "TQ_P22_PREFILL_CAPTURE_LAYER", ".layers.63.self_attn.attn"
    )
    min_cached_len = int(
        os.environ.get("TQ_P22_PREFILL_CAPTURE_MIN_CACHED_LEN", "30000")
    )
    max_captures = int(os.environ.get("TQ_P22_PREFILL_CAPTURE_MAX", "1"))
    captured = 0

    from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl

    original = TurboQuantAttentionImpl._continuation_prefill

    def capture(
        self,
        layer,
        query,
        key_chunk,
        val_chunk,
        kv_cache,
        block_table,
        cached_len,
        seq_len,
        Pi,
        centroids,
    ):
        nonlocal captured
        selected = (
            captured < max_captures
            and layer_filter in layer.layer_name
            and int(cached_len) >= min_cached_len
        )
        snapshot = None
        if selected:
            compact, compact_table = _compact_cache(kv_cache, block_table, seq_len)
            snapshot = dict(
                query=query.detach().cpu(),
                key_chunk=key_chunk.detach().cpu(),
                value_chunk=val_chunk.detach().cpu(),
                kv_cache=compact.cpu(),
                block_table=compact_table.cpu(),
                cached_len=int(cached_len),
                seq_len=int(seq_len),
                scale=float(self.scale),
                layer=layer.layer_name,
                block_size=int(kv_cache.shape[1]),
                cache_shape=list(kv_cache.shape),
                cache_stride=list(kv_cache.stride()),
                original_block_table=block_table.detach().cpu(),
                query_dtype=str(query.dtype),
            )
        result = original(
            self,
            layer,
            query,
            key_chunk,
            val_chunk,
            kv_cache,
            block_table,
            cached_len,
            seq_len,
            Pi,
            centroids,
        )
        if snapshot is not None:
            snapshot["baseline_output"] = result.detach().cpu()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(snapshot, output_path)
            captured += 1
        return result

    TurboQuantAttentionImpl._continuation_prefill = capture


install()
