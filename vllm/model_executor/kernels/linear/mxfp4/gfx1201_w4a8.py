# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in gfx1201 MXFP4/W4A8 reference and decode backend.

Implementation plan: docs/design/gfx1201_radiance_selective_port.md, workstream A.

The reference path is independent of the Triton decode kernel.  The decode
backend is selected only by its explicit environment opt-in and strict shape
and dtype checks.
"""

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.model_executor.kernels.linear.mxfp4.base import (
    MxFp4LinearKernel,
    MxFp4LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mxfp4.emulation import (
    EmulationMxfp4LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_quark

RADIANCE_REFERENCE_COMMIT: Final = "adf9e1f1c9529dd6c971b223a961833376dbd524"
MXFP4_GROUP_SIZE: Final = 32
FP8_E4M3_MAX: Final = 448.0
_E2M1_VALUES: Final = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@triton.jit
def _e2m1_to_fp32(nibble):
    magnitude = nibble & 0x07
    sign = (nibble >> 3) & 1
    bits = 0x3F000000 + (magnitude.to(tl.int32) << 22)
    value = bits.to(tl.float32, bitcast=True)
    value = tl.where(magnitude == 0, 0.0, value)
    value = tl.where(magnitude == 1, 0.5, value)
    return tl.where(sign == 1, -value, value)


def _on_gfx1201() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1201

    return on_gfx1201()


@triton.jit
def _gfx1201_w4a8_decode_kernel(
    x_ptr,
    x_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    out_ptr,
    m,
    n,
    k,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(BLOCK_K % 32 == 0)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    offs_k_packed = tl.arange(0, BLOCK_K // 2).to(tl.int64)
    offs_k_scale = tl.arange(0, BLOCK_K // 32).to(tl.int64)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(k, BLOCK_K)):
        k_start = k_block * BLOCK_K
        x = tl.load(
            x_ptr
            + offs_m[:, None] * stride_xm
            + (k_start + offs_k[None, :]) * stride_xk,
            mask=(offs_m[:, None] < m) & (k_start + offs_k[None, :] < k),
            other=0.0,
        ).to(tl.bfloat16)
        packed = tl.load(
            weight_ptr
            + offs_n[:, None] * stride_wn
            + (k_start // 2 + offs_k_packed[None, :]) * stride_wk,
            mask=(offs_n[:, None] < n)
            & (k_start // 2 + offs_k_packed[None, :] < k // 2),
            other=0,
        )
        low = _e2m1_to_fp32(packed & 0x0F)
        high = _e2m1_to_fp32((packed >> 4) & 0x0F)
        raw_scale = tl.load(
            weight_scale_ptr
            + offs_n[:, None] * stride_sn
            + (k_start // 32 + offs_k_scale[None, :]) * stride_sk,
            mask=(offs_n[:, None] < n)
            & (k_start // 32 + offs_k_scale[None, :] < k // 32),
            other=0,
        )
        scale_bits = raw_scale.to(tl.int32) << 23
        scale_bits = tl.where(raw_scale == 0, 0x00400000, scale_bits)
        scale_bits = tl.where(raw_scale == 255, 0x7FC00000, scale_bits)
        scale = scale_bits.to(tl.float32, bitcast=True)
        scale = tl.reshape(
            tl.broadcast_to(scale[:, :, None], (BLOCK_N, BLOCK_K // 32, 32)),
            (BLOCK_N, BLOCK_K),
        )
        decoded = tl.interleave(low, high)
        weight = tl.trans((decoded * scale).to(tl.bfloat16))
        accumulator = tl.dot(x, weight, acc=accumulator)
    row_scale = tl.load(x_scale_ptr + offs_m, mask=offs_m < m, other=0.0)
    output = (accumulator * row_scale[:, None]).to(tl.bfloat16)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        output,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


@dataclass(frozen=True, slots=True)
class Gfx1201W4A8Contract:
    """Planned tensor/dispatch contract; not a runtime configuration yet."""

    group_size: int = MXFP4_GROUP_SIZE
    output_dtype: torch.dtype = torch.bfloat16
    accumulation_dtype: torch.dtype = torch.float32
    supports_bias: bool = False


def is_gfx1201_w4a8_candidate(*, x: torch.Tensor, bias: torch.Tensor | None) -> bool:
    """Return whether an activation is eligible for the decode prototype."""

    if (
        bias is not None
        or x.ndim < 2
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        return False
    k = x.shape[-1]
    rows = x.numel() // k if k else 0
    return 0 < rows <= 4 and k % MXFP4_GROUP_SIZE == 0


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
    """Launch the row-major small-M W4A8 decode kernel."""

    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise TypeError("gfx1201 W4A8 decode expects a 2D BF16 activation")
    if packed_weight.ndim != 2 or packed_weight.dtype != torch.uint8:
        raise TypeError("gfx1201 W4A8 decode expects packed uint8 weights")
    if weight_scale.ndim != 2 or weight_scale.dtype != torch.uint8:
        raise TypeError("gfx1201 W4A8 decode expects uint8 weight scales")

    m, k = x.shape
    n, packed_k = packed_weight.shape
    if k != packed_k * 2:
        raise ValueError(
            f"activation K={k} does not match packed weight K={packed_k * 2}"
        )
    if k % MXFP4_GROUP_SIZE != 0:
        raise ValueError(f"logical K={k} must be divisible by {MXFP4_GROUP_SIZE}")
    if tuple(weight_scale.shape) != (n, k // MXFP4_GROUP_SIZE):
        raise ValueError(
            "weight scales must have shape "
            f"{(n, k // MXFP4_GROUP_SIZE)}, got {tuple(weight_scale.shape)}"
        )
    if not (x.device == packed_weight.device == weight_scale.device):
        raise ValueError("activation, weights, and scales must be on the same device")

    quantized_x, activation_scale = quantize_activation_fp8_reference(x)
    quantized_x = quantized_x.contiguous()
    activation_scale = activation_scale.contiguous()
    packed_weight = packed_weight.contiguous()
    weight_scale = weight_scale.contiguous()
    output = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _gfx1201_w4a8_decode_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
        quantized_x,
        activation_scale,
        packed_weight,
        weight_scale,
        output,
        m,
        n,
        k,
        *quantized_x.stride(),
        *packed_weight.stride(),
        *weight_scale.stride(),
        *output.stride(),
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=128,
        num_warps=4,
        num_stages=1,
    )
    return output


class Gfx1201Mxfp4W4A8LinearKernel(MxFp4LinearKernel):
    """Opt-in small-M W4A8 decode backend for ROCm gfx1201."""

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.emulation = EmulationMxfp4LinearKernel(config)

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not envs.VLLM_ROCM_USE_GFX1201_MXFP4_W4A8:
            return False, "VLLM_ROCM_USE_GFX1201_MXFP4_W4A8 is not enabled"
        if not _on_gfx1201():
            return False, "only supports ROCm gfx1201"
        if not has_quark():
            return False, "requires amd-quark for the fallback MXFP4 path"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key != kMxfp4Dynamic:
            return False, "only supports dynamic MXFP4 activation metadata"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.emulation.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not is_gfx1201_w4a8_candidate(x=x, bias=bias):
            return self.emulation.apply_weights(layer, x, bias)

        rows = x.numel() // x.shape[-1]
        x_2d = x.reshape(rows, x.shape[-1])
        output = launch_gfx1201_w4a8_decode(
            x_2d,
            layer.weight,
            layer.weight_scale,
        )
        return output.reshape(*x.shape[:-1], layer.weight.shape[0])


def launch_gfx1201_w4a8_prefill(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Planned large-M W4A8 launch entry point (A5)."""

    del x, packed_weight, weight_scale
    raise NotImplementedError("gfx1201 W4A8 prefill kernel is not implemented")
