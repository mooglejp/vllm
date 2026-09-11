# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1201 TurboQuant K8/V4 decode implementation scaffold.

This module is intentionally inert and is NOT imported by production code yet.
It pins the implementation location/API for the work described in
``docs/design/turboquant_gfx1201_k8v4_mtp.md``.

Target profile only:
  * ROCm gfx1201
  * HEAD_SIZE = 256
  * GQA group size = 6
  * FP8 keys (TurboQuant K8)
  * 4-bit values
  * existing TurboQuant SoA cache layout

Implementation order is mandatory:
  1. single-token fused decode using the public launcher below;
  2. gfx1201 tuning;
  3. extend the SAME launcher contract to query_start_loc-aware MTP.

Do not add MSE-key/Pi/centroid branches here. Unsupported profiles must remain
on the existing TurboQuant safe paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# Exact specialization constants. Keep these compile-time in the first kernel.
TARGET_HEAD_SIZE = 256
TARGET_GQA_GROUP_SIZE = 6
TARGET_VALUE_QUANT_BITS = 4

# Existing TurboQuant SoA K8/V4 D=256 layout.
KEY_DATA_BYTES = 256
VALUE_DATA_BYTES = 128
DATA_BYTES_PER_SLOT = KEY_DATA_BYTES + VALUE_DATA_BYTES  # 384
NUM_SOA_FIELDS = 2
SOA_V_SCALE = 0
SOA_V_ZERO = 1
LOGICAL_BYTES_PER_SLOT = DATA_BYTES_PER_SLOT + NUM_SOA_FIELDS * 2  # 388


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
    """Public launcher placeholder for the gfx1201 K8/V4 fast path.

    Phase 2 implementation contract:
      * initially require one query token per request, but accept
        ``query_start_loc`` now so MTP does not require an API redesign;
      * consume the existing SoA layout exactly;
      * load FP8 K directly, unpack V4 once per packed byte, and fuse QK,
        online softmax and P*V in stage 1;
      * use reusable caller workspace for split-K and graph stability;
      * never allocate ``torch.arange(B + 1)`` in the hot path.

    Phase 5 extension:
      * remove the one-token restriction;
      * map tokens to requests using ``query_start_loc``;
      * apply per-query causal limits using ``context_len + q_pos``;
      * share each K/V tile across query rows from the same request/GQA group.

    The function must remain unreachable from production dispatch until its
    correctness gates in the design document pass.
    """
    raise NotImplementedError(
        "gfx1201 K8/V4 fused decode is an implementation placeholder; "
        "follow docs/design/turboquant_gfx1201_k8v4_mtp.md"
    )


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
) -> None:
    """Phase 2 kernel placeholder.

    Initial shape: grid over (request, kv_head, split), one program owning the
    six Q heads for a KV head. Start with BLOCK_M=16, TILE_SIZE=16,
    num_stages=1, fp32 online-softmax accumulators. Tune only after correctness.
    """
    raise NotImplementedError


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
    """Phase 5 MTP kernel placeholder.

    This must not treat query.shape[0] as the number of requests. It must use
    query_start_loc to derive each request's q_len and causal query position.
    Workspace indexing is per query token, while block_table/seq_lens indexing
    is per request.
    """
    raise NotImplementedError
