# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eligibility contract for the gfx1201 TurboQuant K8/V4 fast path.

The single- and multi-token routes are wired in ``turboquant_attn.py`` behind
an explicit opt-in. This module keeps the target-profile decision separate
from the kernel launchers and cache implementation.
"""

from __future__ import annotations

TARGET_ARCH = "gfx1201"
TARGET_HEAD_SIZE = 256
TARGET_GQA_GROUP_SIZE = 6
TARGET_VALUE_QUANT_BITS = 4
INITIAL_SUPPORTED_BLOCK_SIZES = (16, 32)


def is_target_profile(
    *,
    rocm_arch: str,
    head_size: int,
    num_kv_groups: int,
    key_fp8: bool,
    value_quant_bits: int,
    block_size: int,
    has_sinks: bool,
    sliding_window: int | None,
) -> bool:
    """Return whether the configuration matches the supported target profile."""
    return (
        rocm_arch == TARGET_ARCH
        and head_size == TARGET_HEAD_SIZE
        and num_kv_groups == TARGET_GQA_GROUP_SIZE
        and key_fp8
        and value_quant_bits == TARGET_VALUE_QUANT_BITS
        and block_size in INITIAL_SUPPORTED_BLOCK_SIZES
        and not has_sinks
        and not (sliding_window and sliding_window > 0)
    )
