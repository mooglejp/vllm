# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eligibility contract for the gfx1201 TurboQuant K8/V4 fast path.

The single-token route is wired in ``turboquant_attn.py`` behind an explicit
opt-in. The multi-token/MTP route remains intentionally disabled until its
query-start-location and workspace contracts are implemented.
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
    """Strict initial eligibility contract for the gfx1201 fast route.

    Keep this deliberately narrow until the R9700 implementation is proven.
    Production wiring should make this decision once during attention/backend
    initialization so cache layout cannot change after KV population begins.
    """
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


# ---------------------------------------------------------------------------
# Remaining MTP integration anchors in turboquant_attn.py
# ---------------------------------------------------------------------------
#
# Phase 3 -- after the single-token kernel is correct:
#
# TurboQuantAttentionImpl.__init__
#   * detect exact gfx1201 target profile once;
#   * choose fast-path enablement once;
#   * when fast mode is enabled, choose SoA store BEFORE the first KV write;
#   * never change that layout decision later.
#
# TurboQuantAttentionImpl._decode_attention
#   * dispatch the dedicated gfx1201 launcher directly;
#   * pass attn_metadata.query_start_loc (no per-call torch.arange);
#   * if this layer selected SoA but the dedicated kernel is unavailable for
#     the current feature, fall back to _dispatch_decode_soa(), never AoS.
#
# TurboQuantAttentionImpl._store_kv
#   * no new layout: reuse the existing SoA store exactly.
#
# TurboQuantAttentionImpl._prefill_attention
#   * continuation reads must match the selected cache layout;
#   * do not send continuation/MTP-shaped queries to the fast kernel until the
#     multi-token causal path has passed its correctness matrix.
#
# Phase 5 -- only after native multi-token decode is implemented:
#
# TurboQuantMetadataBuilder.__init__
#   * enable supports_spec_as_decode=True only for configurations whose decode
#     path is known to support query_len > 1 correctly.
#
# TurboQuantMetadataBuilder._reserve_workspace
#   * size partial/output workspace by maximum decode QUERY TOKENS, not merely
#     maximum request count. MTP makes num_decode_tokens > num_decodes.
#
# forward/_decode_attention audit
#   * num_decodes is a request count;
#   * num_decode_tokens is a token count and is the tensor split dimension.
#
# Do not delete these notes until the corresponding design-document phases are
# implemented and covered by tests.
