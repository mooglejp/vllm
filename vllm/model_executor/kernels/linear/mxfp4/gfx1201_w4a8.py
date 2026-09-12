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
import torch.nn.functional as F

RADIANCE_REFERENCE_COMMIT: Final = "adf9e1f1c9529dd6c971b223a961833376dbd524"
MXFP4_GROUP_SIZE: Final = 32
FP8_E4M3_MAX: Final = 448.0
_E2M1_VALUES: Final = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


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


def quantize_activation_fp8_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 rows to FP8 E4M3 with one scale per row.

    The scale is the row maximum divided by the finite FP8 maximum. An all-zero
    row uses a scale of one so it remains finite and round-trippable. The
    returned scale is the dequantization scale, rather than its reciprocal.

    This is a reference-only implementation of the A2 contract. It is kept
    independent of the HIP path so the first kernel can be checked against it.
    """

    if x.ndim != 2:
        raise ValueError(f"expected a 2D activation tensor, got shape {x.shape}")
    if not x.is_floating_point():
        raise TypeError(f"expected a floating-point activation tensor, got {x.dtype}")

    x_float = x.to(torch.float32)
    row_amax = x_float.abs().amax(dim=-1)
    scale = torch.where(
        row_amax == 0, torch.ones_like(row_amax), row_amax / FP8_E4M3_MAX
    )
    normalized = (x_float / scale[:, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return normalized.to(torch.float8_e4m3fn), scale


def dequantize_fp8_activation_reference(
    quantized: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize the row-scaled FP8 activation produced by the reference."""

    if quantized.ndim != 2:
        raise ValueError(
            f"expected a 2D quantized activation tensor, got {quantized.shape}"
        )
    if quantized.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected float8_e4m3fn activations, got {quantized.dtype}")
    if scale.ndim != 1 or scale.shape[0] != quantized.shape[0]:
        raise ValueError(
            "activation scale must have shape [M] matching the quantized rows"
        )

    return quantized.to(torch.float32) * scale.to(torch.float32)[:, None]


def dequantize_mxfp4_weight_reference(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode row-major OCP MXFP4 weights with E8M0 group scales.

    ``packed_weight`` stores two E2M1 codewords per byte, with the low nibble
    first. ``weight_scale`` stores one E8M0 exponent byte for each group of 32
    logical K values. The implementation intentionally uses only tensor
    operations and does not call a vLLM or HIP decode helper.
    """

    if packed_weight.ndim != 2 or packed_weight.dtype != torch.uint8:
        raise TypeError("packed MXFP4 weights must be a 2D uint8 tensor")
    if weight_scale.ndim != 2 or weight_scale.dtype != torch.uint8:
        raise TypeError("MXFP4 weight scales must be a 2D uint8 tensor")
    if not dtype.is_floating_point:
        raise TypeError(f"decoded weights require a floating-point dtype, got {dtype}")

    n, packed_k = packed_weight.shape
    k = packed_k * 2
    if k % MXFP4_GROUP_SIZE != 0:
        raise ValueError(f"logical K={k} must be divisible by {MXFP4_GROUP_SIZE}")
    expected_scale_shape = (n, k // MXFP4_GROUP_SIZE)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"expected MXFP4 scales with shape {expected_scale_shape}, "
            f"got {tuple(weight_scale.shape)}"
        )

    low = packed_weight & 0x0F
    high = (packed_weight >> 4) & 0x0F
    codewords = torch.stack((low, high), dim=-1).flatten(-2)
    magnitudes = codewords & 0x07
    signs = torch.where(codewords < 8, 1.0, -1.0)
    values = torch.tensor(_E2M1_VALUES, device=packed_weight.device)
    decoded = values[magnitudes.long()] * signs

    scale_exponents = weight_scale.to(torch.int16) - 127
    scales = torch.exp2(scale_exponents.to(torch.float32))
    scales = torch.where(weight_scale == 255, torch.nan, scales)
    scales = scales.repeat_interleave(MXFP4_GROUP_SIZE, dim=-1)
    return (decoded * scales).to(dtype)


def gfx1201_w4a8_linear_reference(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Run the independent BF16-input W4A8 reference linear operation.

    Activation quantization is row-scaled FP8 E4M3. MXFP4 weights are
    dequantized to FP32, accumulation is performed by ``F.linear`` in FP32,
    and the result is returned as BF16 as required by the planned backend
    contract.
    """

    if x.ndim != 2:
        raise ValueError(f"expected a 2D activation tensor, got shape {x.shape}")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"the W4A8 contract requires BF16 activations, got {x.dtype}")
    if x.shape[-1] != packed_weight.shape[-1] * 2:
        raise ValueError(
            f"activation K={x.shape[-1]} does not match packed weight K="
            f"{packed_weight.shape[-1] * 2}"
        )

    quantized_x, activation_scale = quantize_activation_fp8_reference(x)
    dequantized_x = dequantize_fp8_activation_reference(quantized_x, activation_scale)
    dequantized_weight = dequantize_mxfp4_weight_reference(packed_weight, weight_scale)
    return F.linear(dequantized_x, dequantized_weight).to(torch.bfloat16)


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
