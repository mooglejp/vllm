# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inert scaffold for a gfx1201 MTP draft-only INT2 exact-rerank head.

Implementation plan: docs/design/gfx1201_radiance_selective_port.md, workstream B.

Nothing imports this module.  The target logits processor and shared BF16
lm-head must remain untouched until the capture/recall gate has passed.
"""

from dataclasses import dataclass
from typing import Final, NamedTuple

import torch

RADIANCE_REFERENCE_COMMIT: Final = "adf9e1f1c9529dd6c971b223a961833376dbd524"
DRAFT_WEIGHT_BITS: Final = 2
DRAFT_GROUP_SIZE: Final = 128
DRAFT_BLOCK_N: Final = 64
INITIAL_CANDIDATES_PER_BLOCK: Final = 8


@dataclass(frozen=True, slots=True)
class Gfx1201FastDraftHeadContract:
    """Initial bounded search space for workstream B."""

    bits: int = DRAFT_WEIGHT_BITS
    group_size: int = DRAFT_GROUP_SIZE
    block_n: int = DRAFT_BLOCK_N
    candidates_per_block: int = INITIAL_CANDIDATES_PER_BLOCK
    rerank_widths: tuple[int, ...] = (32, 64)


class PackedDraftHead(NamedTuple):
    """Planned auxiliary state; BF16 target/draft shared weight is kept separately."""

    packed_weight: torch.Tensor
    scale: torch.Tensor
    zero_term: torch.Tensor
    vocab_size: int
    hidden_size: int


def is_fast_draft_head_candidate(*, hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    """Return False until B1/B2 are implemented and the opt-in is wired."""

    del hidden_states, weight
    return False


def pack_int2_draft_head(weight: torch.Tensor) -> PackedDraftHead:
    """Planned asymmetric INT2-g128 pack operation.

    B1: implement in row chunks so no full-head FP32 temporary remains allocated.
    The original BF16 `weight` must remain resident for target verification and
    exact candidate reranking.
    """

    del weight
    raise NotImplementedError("gfx1201 INT2 draft-head packing is not implemented")


def coarse_draft_candidates(
    hidden_states: torch.Tensor,
    packed: PackedDraftHead,
    *,
    candidates_per_block: int = INITIAL_CANDIDATES_PER_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Planned coarse candidate values/ids for B1/B2."""

    del hidden_states, packed, candidates_per_block
    raise NotImplementedError("gfx1201 draft-head coarse kernel is not implemented")


def exact_rerank_argmax(
    hidden_states: torch.Tensor,
    bf16_weight: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Planned exact BF16 rerank returning one proposed token per row."""

    del hidden_states, bf16_weight, candidate_ids
    raise NotImplementedError("gfx1201 draft-head exact rerank is not implemented")
