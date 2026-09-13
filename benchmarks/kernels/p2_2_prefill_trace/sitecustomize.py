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
from torch.nn.attention import SDPBackend, sdpa_kernel

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
    _, soa_dequant, _ = turboquant_attn._soa_imports()

    def capture_prefix(
        self,
        layer,
        query,
        key_chunk,
        kv_cache,
        block_table,
        cached_len,
        seq_len,
        centroids,
    ):
        if not self._soa_store:
            raise RuntimeError("the baseline snapshot requires the SoA cache")
        q_len, _, head_dim = query.shape
        num_kv_heads = key_chunk.shape[1]
        block_size = int(kv_cache.shape[1])
        alloc_len = math.ceil(int(cached_len) / block_size) * block_size
        k_cached = torch.empty(
            1,
            num_kv_heads,
            alloc_len,
            head_dim,
            dtype=torch.float16,
            device=query.device,
        )
        v_cached = torch.empty_like(k_cached)
        key_fp8 = bool(self.tq_config.key_fp8)
        mse_bytes = self._mse_bytes
        val_data_bytes = self._val_data_bytes
        key_data_bytes = head_dim if key_fp8 else mse_bytes
        data_bytes_per_slot = key_data_bytes + val_data_bytes
        meta_region_offset = block_size * num_kv_heads * data_bytes_per_slot
        num_soa_fields = 2 if key_fp8 else 3
        soa_dequant[(alloc_len, num_kv_heads)](
            kv_cache,
            kv_cache.view(torch.uint16),
            block_table,
            centroids,
            k_cached,
            v_cached,
            k_cached.stride(0),
            k_cached.stride(1),
            k_cached.stride(2),
            v_cached.stride(0),
            v_cached.stride(1),
            v_cached.stride(2),
            kv_cache.stride(0),
            block_table.stride(0),
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=num_kv_heads,
            MSE_BYTES=mse_bytes,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            MSE_BITS=self.tq_config.key_mse_bits,
            KEY_FP8=1 if key_fp8 else 0,
            KEY_DATA_BYTES=key_data_bytes,
            META_REGION_OFFSET=meta_region_offset,
            NUM_SOA_FIELDS=num_soa_fields,
            SOA_K_NORM=0,
            SOA_V_SCALE=0 if key_fp8 else 1,
            SOA_V_ZERO=1 if key_fp8 else 2,
            BLOCK_D=1 << (head_dim - 1).bit_length(),
            NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
            FP8_E4B15=turboquant_attn._use_fp8_e4b15(query.device.index or 0),
            num_warps=4,
        )
        if key_fp8:
            prefix_key = k_cached[0, :, :cached_len, :].transpose(0, 1)
        else:
            Pi_half = layer._tq_Pi_half
            k_flat = k_cached[0, :, :cached_len, :].reshape(-1, head_dim)
            prefix_key = (
                (k_flat @ Pi_half)
                .reshape(num_kv_heads, cached_len, head_dim)
                .transpose(0, 1)
            )
        prefix_value = v_cached[0, :, :cached_len, :].transpose(0, 1)
        if prefix_key.shape[0] != cached_len or seq_len != cached_len + q_len:
            raise RuntimeError("baseline snapshot contract shape mismatch")
        return prefix_key.contiguous(), prefix_value.contiguous()

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
        with sdpa_kernel(SDPBackend.MATH):
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
            "compared_input_fields": [
                "query",
                "key_chunk",
                "value_chunk",
            ],
            "uncompared_input_fields": [
                "prefix_key",
                "prefix_value",
            ],
            "sdpa_backend": "MATH",
            "scale": float(self.scale),
            "causal_boundary": {
                "cached_len": int(cached_len),
                "q_len": int(query.shape[0]),
                "seq_len": int(seq_len),
            },
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
            prefix_key_fp16, prefix_value_fp16 = capture_prefix(
                self,
                layer,
                query,
                key_chunk,
                kv_cache,
                block_table,
                int(cached_len),
                int(seq_len),
                centroids,
            )
            compact, compact_table = _compact_cache(kv_cache, block_table, int(seq_len))
            q_positions = torch.arange(int(cached_len), int(seq_len), dtype=torch.int64)
            k_positions = torch.arange(int(seq_len), dtype=torch.int64)
            causal_mask = k_positions[None, :] <= q_positions[:, None]
            prefix_key_bf16 = prefix_key_fp16.to(query.dtype)
            prefix_value_bf16 = prefix_value_fp16.to(query.dtype)
            torch.save(
                {
                    "query": query.detach().cpu(),
                    "key_chunk": key_chunk.detach().cpu(),
                    "value_chunk": val_chunk.detach().cpu(),
                    "prefix_key_fp16": prefix_key_fp16.detach().cpu(),
                    "prefix_value_fp16": prefix_value_fp16.detach().cpu(),
                    "prefix_key_bf16": prefix_key_bf16.detach().cpu(),
                    "prefix_value_bf16": prefix_value_bf16.detach().cpu(),
                    "kv_cache": compact.cpu(),
                    "block_table": compact_table.cpu(),
                    "cached_len": int(cached_len),
                    "seq_len": int(seq_len),
                    "q_positions": q_positions,
                    "k_positions": k_positions,
                    "causal_mask": causal_mask,
                    "scale": float(self.scale),
                    "sdpa_backend": "MATH",
                    "sdpa_math_forced": True,
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
