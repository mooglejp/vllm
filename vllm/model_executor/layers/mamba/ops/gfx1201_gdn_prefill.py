# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inert scaffold for profile-gated gfx1201 GDN prefill fusion.

Plan: docs/design/gfx1201_radiance_long_prefill.md, phase P5.

Nothing imports this module.  Do not implement or connect it unless the
post-attention/linear prefill profile shows GDN/FLA is a material cost center.
"""

from dataclasses import dataclass
from typing import Final

import torch

REFERENCE_HEAD_K: Final = 128
REFERENCE_HEAD_V: Final = 128
REFERENCE_CHUNK: Final = 64


@dataclass(frozen=True, slots=True)
class Gfx1201GdnPrefillContract:
    """Reference geometry to verify against the live model before P5."""

    head_k: int = REFERENCE_HEAD_K
    head_v: int = REFERENCE_HEAD_V
    chunk: int = REFERENCE_CHUNK


def is_gfx1201_gdn_prefill_candidate(*, q: torch.Tensor, v: torch.Tensor) -> bool:
    """Remain false until P5 profile and correctness gates are complete."""

    del q, v
    return False


def launch_gfx1201_gdn_prefill(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Planned fused GDN prefill entry point returning output and final state."""

    del args, kwargs
    raise NotImplementedError("gfx1201 GDN prefill fusion is not implemented")
