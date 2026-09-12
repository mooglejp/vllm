# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm.model_executor.kernels.linear.mxfp4.base import (
    MxFp4LinearKernel,
    MxFp4LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mxfp4.emulation import (
    EmulationMxfp4LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_quark

_MAX_FUSED_ROWS = 4
_MIN_FUSED_OUTPUT_SIZE = 512


def _on_gfx1201() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1201

    return on_gfx1201()


@triton.jit
def _e2m1_to_fp32(nibble):
    magnitude = nibble & 0x07
    sign = (nibble >> 3) & 1
    bits = 0x3F000000 + (magnitude.to(tl.int32) << 22)
    value = bits.to(tl.float32, bitcast=True)
    value = tl.where(magnitude == 0, 0.0, value)
    value = tl.where(magnitude == 1, 0.5, value)
    return tl.where(sign == 1, -value, value)


@triton.jit
def _mxfp4_small_m_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
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
        )
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
        decoded = tl.interleave(low, high)

        raw_scale = tl.load(
            scale_ptr
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
        decoded_weight = tl.trans((decoded * scale).to(tl.bfloat16))
        accumulator = tl.dot(x, decoded_weight, acc=accumulator)

    out = accumulator.to(out_ptr.type.element_ty)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


def triton_mxfp4_small_m_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Multiply BF16 rows by row-major OCP MXFP4 weights on gfx1201."""
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _mxfp4_small_m_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
        x,
        weight,
        weight_scale,
        out,
        m,
        n,
        k,
        *x.stride(),
        *weight.stride(),
        *weight_scale.stride(),
        *out.stride(),
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=128,
        num_warps=4,
        num_stages=1,
    )
    return out


class TritonGfx1201Mxfp4LinearKernel(MxFp4LinearKernel):
    """Software-fused OCP MXFP4 small-M GEMM for gfx1201."""

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.emulation = EmulationMxfp4LinearKernel(config)

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not envs.VLLM_ROCM_USE_GFX1201_MXFP4_GEMM:
            return False, "VLLM_ROCM_USE_GFX1201_MXFP4_GEMM is not enabled"
        if not _on_gfx1201():
            return False, "only supports ROCm gfx1201"
        if not has_quark():
            return False, "requires amd-quark for activation MXFP4 QDQ"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key != kMxfp4Dynamic:
            return False, "only supports dynamic MXFP4 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.emulation.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        output_size = layer.weight.shape[0]
        if (
            rows > _MAX_FUSED_ROWS
            or output_size < _MIN_FUSED_OUTPUT_SIZE
            or x.dtype != torch.bfloat16
            or bias is not None
        ):
            return self.emulation.apply_weights(layer, x, bias)

        quantized_x = self.emulation.quant_dequant_func(x)
        output = triton_mxfp4_small_m_linear(
            quantized_x.reshape(rows, x.shape[-1]),
            layer.weight,
            layer.weight_scale,
        )
        return output.reshape(*x.shape[:-1], output_size)
