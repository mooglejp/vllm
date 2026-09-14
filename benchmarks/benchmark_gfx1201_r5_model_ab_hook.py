# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a benchmark-only R5 continuation attention hook.

This module is injected with ``PYTHONPATH`` into an isolated model process. It
does not edit or register a production backend.  Only
``TurboQuantAttentionImpl._continuation_prefill`` calls with
``cached_len > 0`` and ``q_len > 128`` temporarily route their existing
materialization through the loaded AMD Triton FlashAttention function. All
other calls invoke the original method unchanged.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from pathlib import Path
from typing import Any

import flash_attn
import torch

from vllm.model_executor.warmup import kernel_warmup
from vllm.v1.attention.backends import turboquant_attn
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl
from vllm.v1.worker.gpu import warmup

_ORIGINAL = TurboQuantAttentionImpl._continuation_prefill
_ORIGINAL_FORWARD = TurboQuantAttentionImpl.forward
# Match the previously validated diagnostic Math baseline in both arms.
warmup.warmup_kernels = lambda *args, **kwargs: None
kernel_warmup.kernel_warmup = lambda *args, **kwargs: None
turboquant_attn._HAS_FLASH_ATTN = False
_LOCK = threading.Lock()
_COUNTS: dict[str, Any] = {
    "enabled": os.environ.get("R5_MODE") == "candidate",
    "calls": 0,
    "shapes": {},
    "layers": {},
    "applied_calls": 0,
    "restored_globals": True,
}


def _record(cached_len: int, q_len: int) -> None:
    with _LOCK:
        _COUNTS["calls"] += 1
        key = f"cached{cached_len}_q{q_len}"
        shapes = _COUNTS["shapes"]
        shapes[key] = int(shapes.get(key, 0)) + 1


def _refresh_control() -> None:
    control_path = os.environ.get("R5_MODE_FILE")
    if not control_path or not Path(control_path).exists():
        return
    control = json.loads(Path(control_path).read_text())
    mode = control["mode"]
    if mode not in {"baseline", "candidate"}:
        raise ValueError(f"Invalid diagnostic mode: {mode}")
    if _COUNTS.get("run_id") != control["run_id"]:
        _COUNTS.update(calls=0, applied_calls=0, shapes={}, layers={})
        _COUNTS["enabled"] = mode == "candidate"
        _COUNTS["run_id"] = control["run_id"]
        _write_stats()
    elif _COUNTS["enabled"] != (mode == "candidate"):
        raise RuntimeError("Diagnostic mode changed within a run ID")


def _forward(self: TurboQuantAttentionImpl, *args: Any, **kwargs: Any):
    # First chunks must reset counters even without eligible continuation.
    _refresh_control()
    return _ORIGINAL_FORWARD(self, *args, **kwargs)


def _patched(self: TurboQuantAttentionImpl, *args: Any, **kwargs: Any):
    query = kwargs.get("query", args[1] if len(args) > 1 else None)
    cached_len = kwargs.get("cached_len", args[6] if len(args) > 6 else None)
    q_len = int(query.shape[0]) if query is not None else 0
    cached_len = int(cached_len) if cached_len is not None else 0
    layer = kwargs.get("layer", args[0] if args else None)
    layer_name = getattr(layer, "layer_name", "")
    target = layer_name.startswith("language_model.model.layers.")
    if not target or cached_len <= 0 or q_len <= 128:
        return _ORIGINAL(self, *args, **kwargs)

    _refresh_control()
    _record(cached_len, q_len)
    _COUNTS["layers"][layer_name] = _COUNTS["layers"].get(layer_name, 0) + 1
    if _COUNTS["enabled"]:
        _COUNTS["applied_calls"] += 1
    if ".layers.63." in layer_name:
        _write_stats()
    if not _COUNTS["enabled"]:
        with torch.profiler.record_function("r5_baseline_continuation"):
            return _ORIGINAL(self, *args, **kwargs)
    old_has = turboquant_attn._HAS_FLASH_ATTN
    old_func = getattr(turboquant_attn, "flash_attn_varlen_func", None)
    turboquant_attn._HAS_FLASH_ATTN = True
    turboquant_attn.flash_attn_varlen_func = _upstream_varlen
    try:
        with torch.profiler.record_function("r5_candidate_continuation"):
            return _ORIGINAL(self, *args, **kwargs)
    finally:
        turboquant_attn._HAS_FLASH_ATTN = old_has
        if old_func is None:
            turboquant_attn.__dict__.pop("flash_attn_varlen_func", None)
        else:
            turboquant_attn.flash_attn_varlen_func = old_func


TurboQuantAttentionImpl._continuation_prefill = _patched
TurboQuantAttentionImpl.forward = _forward


def _upstream_varlen(*args: Any, **kwargs: Any):
    version = kwargs.pop("fa_version", None)
    if version not in (None, 2):
        raise ValueError(f"Unexpected FlashAttention API version: {version}")
    return flash_attn.flash_attn_varlen_func(*args, **kwargs)


def _write_stats() -> None:
    path = os.environ.get("R5_HOOK_STATS")
    if not path:
        return
    _COUNTS["peak_allocated_bytes"] = torch.accelerator.memory.max_memory_allocated()
    _COUNTS["peak_reserved_bytes"] = torch.accelerator.memory.max_memory_reserved()
    Path(f"{path}.{os.getpid()}").write_text(json.dumps(_COUNTS, sort_keys=True) + "\n")


atexit.register(_write_stats)
