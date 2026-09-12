# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inert scaffold for gfx1201 TurboQuant K8/V4 continuation prefill.

Plan: docs/design/gfx1201_radiance_long_prefill.md, phase P2.

This module is intentionally not imported.  It defines the first functional
contract only; runtime behavior must remain unchanged until the P2 gates pass.
"""

from dataclasses import dataclass
from typing import Final

import torch

HEAD_DIM: Final = 256
GQA: Final = 6


@dataclass(frozen=True, slots=True)
class Gfx1201TurboQuantPrefillContract:
    """Narrow initial contract for a direct K8/V4 continuation kernel."""

    head_dim: int = HEAD_DIM
    gqa: int = GQA
    causal: bool = True
    supports_sinks: bool = False
    supports_sliding_window: bool = False


def is_gfx1201_tq_prefill_candidate(
    *,
    query: torch.Tensor,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    cached_len: int,
) -> bool:
    """Remain false until P2 installs and validates a production path."""

    del query, key_chunk, value_chunk, cached_len
    return False


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
) -> torch.Tensor:
    """Planned streaming continuation-preFill entry point.

    The adopted P2 implementation must read prefix K/V directly from the
    existing SoA K8/V4 cache, use raw current-chunk K/V, apply the causal bound
    `kv_pos <= cached_len + query_row`, and avoid full-prefix K/V materialization.
    """

    del (
        query,
        key_chunk,
        value_chunk,
        kv_cache,
        block_table,
        cached_len,
        seq_len,
        scale,
    )
    raise NotImplementedError("gfx1201 K8/V4 continuation prefill is not implemented")
