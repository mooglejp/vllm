# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic-only 4K P2.2 continuation trace hook.

The hook records immutable input/output digests for each large-continuation
attention call. Optional target snapshots are saved for offline replay. It is
loaded only from a diagnostic PYTHONPATH and never changes production code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import regex as re
import torch

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.attn$")


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return value.numpy().tobytes()


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _stats(tensor: torch.Tensor) -> dict[str, float]:
    value = tensor.detach().float()
    return {
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "abs_max": float(value.abs().max()),
    }


def _compact_cache(
    cache: torch.Tensor, block_table: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = int(cache.shape[1])
    count = math.ceil(seq_len / block_size)
    physical = block_table[0, :count].to(torch.long)
    slot_bytes = int(cache.shape[-1])
    raw_bytes = block_size * int(cache.shape[2]) * slot_bytes
    raw_blocks = cache.as_strided((cache.shape[0], raw_bytes), (cache.stride(0), 1))
    compact = raw_blocks[physical].reshape(
        count, block_size, int(cache.shape[2]), slot_bytes
    )
    return compact.contiguous(), torch.arange(
        count, dtype=torch.int32, device=cache.device
    )[None]


def _operation(self, layer_name: str, query: torch.Tensor) -> str:
    enabled = os.environ.get("VLLM_TQ_GFX1201_K8V4_PREFILL", "false").lower()
    fast = enabled in {"1", "true", "yes", "on"} and self._use_gfx1201_fast
    if not fast:
        return "sdpa_math_fallback"
    pv = os.environ.get("VLLM_TQ_GFX1201_K8V4_PREFILL_PV_FP32", "false").lower()
    if pv in {"1", "true", "yes", "on"}:
        return "gfx1201_streaming_pv_fp32"
    return "gfx1201_streaming_bf16_pv"


def install() -> None:
    destination = os.environ.get("TQ_P22_PREFILL_TRACE_OUTPUT")
    if not destination:
        return
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_layer = os.environ.get("TQ_P22_PREFILL_TRACE_TARGET_LAYER")
    target_cached = os.environ.get("TQ_P22_PREFILL_TRACE_TARGET_CACHED_LEN")
    snapshot_path = os.environ.get("TQ_P22_PREFILL_TRACE_SNAPSHOT")
    max_seq_len = int(os.environ.get("TQ_P22_PREFILL_TRACE_SEQ_LEN", "4096"))
    cache_digest = os.environ.get("TQ_P22_PREFILL_TRACE_CACHE_HASH") == "1"
    trace_index = 0

    from vllm.model_executor.warmup import kernel_warmup
    from vllm.v1.attention.backends import turboquant_attn
    from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl
    from vllm.v1.worker.gpu import warmup

    warmup.warmup_kernels = lambda *args, **kwargs: None
    kernel_warmup.kernel_warmup = lambda *args, **kwargs: None
    turboquant_attn._HAS_FLASH_ATTN = False
    original = TurboQuantAttentionImpl._continuation_prefill

    def trace(
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
        nonlocal trace_index
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
        match = _LAYER_RE.search(layer.layer_name)
        selected = (
            match is not None and int(cached_len) > 0 and int(seq_len) <= max_seq_len
        )
        if not selected:
            return result
        layer_index = int(match[1])
        record = {
            "trace_index": trace_index,
            "pid": os.getpid(),
            "layer": layer.layer_name,
            "layer_index": layer_index,
            "cached_len": int(cached_len),
            "q_len": int(query.shape[0]),
            "seq_len": int(seq_len),
            "operation": _operation(self, layer.layer_name, query),
            "query_dtype": str(query.dtype),
            "query_shape": list(query.shape),
            "query_digest": _digest(query),
            "key_chunk_digest": _digest(key_chunk),
            "value_chunk_digest": _digest(val_chunk),
            "output_digest": _digest(result),
            "output_stats": _stats(result),
        }
        if cache_digest:
            record["cache_digest"] = _digest(kv_cache)
            record["block_table_digest"] = _digest(block_table)
        with output_path.open("a") as output:
            output.write(json.dumps(record) + "\n")
        target = (
            snapshot_path
            and target_layer is not None
            and target_cached is not None
            and layer_index == int(target_layer)
            and int(cached_len) == int(target_cached)
        )
        if target and not Path(snapshot_path).exists():
            compact, compact_table = _compact_cache(kv_cache, block_table, int(seq_len))
            torch.save(
                {
                    "query": query.detach().cpu(),
                    "key_chunk": key_chunk.detach().cpu(),
                    "value_chunk": val_chunk.detach().cpu(),
                    "kv_cache": compact.cpu(),
                    "block_table": compact_table.cpu(),
                    "cached_len": int(cached_len),
                    "seq_len": int(seq_len),
                    "scale": float(self.scale),
                    "layer": layer.layer_name,
                    "attention_output": result.detach().cpu(),
                    "trace_record": record,
                },
                snapshot_path,
            )
        trace_index += 1
        return result

    TurboQuantAttentionImpl._continuation_prefill = trace


if os.environ.get("TQ_P22_PREFILL_TRACE_ENABLE") == "1":
    install()
