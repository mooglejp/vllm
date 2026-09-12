# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inert scaffold for an opt-in gfx1201 MXFP4/W4A8 linear backend.

Implementation plan: docs/design/gfx1201_radiance_selective_port.md, workstream A.

This module is intentionally not imported or registered.  It must not change
runtime behavior until the numerical reference, provenance/license gate, and
phase-A adoption gates are implemented.
"""

from dataclasses import dataclass
from typing import Final

import torch

RADIANCE_REFERENCE_COMMIT: Final = "adf9e1f1c9529dd6c971b223a961833376dbd524"
MXFP4_GROUP_SIZE: Final = 32
FP8_E4M3_MAX: Final = 448.0


@dataclass(frozen=True, slots=True)
class Gfx1201W4A8Contract:
    """Planned tensor/dispatch contract; not a runtime configuration yet."""

    group_size: int = MXFP4_GROUP_SIZE
    output_dtype: torch.dtype = torch.bfloat16
    accumulation_dtype: torch.dtype = torch.float32
    supports_bias: bool = False


def is_gfx1201_w4a8_candidate(*, x: torch.Tensor, bias: torch.Tensor | None) -> bool:
    """Return False until workstream A installs and validates the backend.

    Luna: replace this inert result only in the functional W4A8 commit, and keep
    architecture/opt-in/config checks in the owning `MxFp4LinearKernel` class.
    """

    del x, bias
    return False


def quantize_activation_fp8_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent reference for the planned BF16 -> FP8 E4M3 activation step.

    R0/A2: define the exact scale/clamp/rounding contract here before writing the
    HIP kernel.  This function is deliberately unavailable in the scaffold so a
    caller cannot accidentally treat an unreviewed formula as production truth.
    """

    raise NotImplementedError("gfx1201 W4A8 activation reference is not implemented")


def permute_mxfp4_weight_for_wmma(weight: torch.Tensor) -> torch.Tensor:
    """Planned one-time fragment-order permutation from workstream A4.

    The first functional W4A8 port must stay in checkpoint row-major order.
    Implement this only after the row-major decode backend passes A3.
    """

    raise NotImplementedError("gfx1201 W4A8 weight permutation is not implemented")


def launch_gfx1201_w4a8_decode(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Planned small-M W4A8 launch entry point (A3)."""

    del x, packed_weight, weight_scale
    raise NotImplementedError("gfx1201 W4A8 decode kernel is not implemented")


def launch_gfx1201_w4a8_prefill(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Planned large-M W4A8 launch entry point (A5)."""

    del x, packed_weight, weight_scale
    raise NotImplementedError("gfx1201 W4A8 prefill kernel is not implemented")
