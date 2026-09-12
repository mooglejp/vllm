# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.mxfp4.triton_gfx1201 import (
    triton_mxfp4_small_m_linear,
)
from vllm.platforms import current_platform


def _on_gfx1201() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1201

    return on_gfx1201()


def _dequantize_reference(
    packed_weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    low = packed_weight & 0x0F
    high = (packed_weight >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).flatten(-2)
    magnitude = nibbles & 0x07
    sign = torch.where(nibbles < 8, 1.0, -1.0)
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed_weight.device,
    )
    decoded = values[magnitude.long()] * sign
    scales = (
        weight_scale.view(torch.float8_e8m0fnu)
        .to(torch.float32)
        .repeat_interleave(32, dim=1)
    )
    return (decoded * scales).to(torch.bfloat16)


@pytest.mark.skipif(not _on_gfx1201(), reason="requires ROCm gfx1201")
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(1, 512, 128), (3, 544, 160), (4, 512, 96)],
)
def test_small_m_matches_dequantize_then_linear(m: int, n: int, k: int):
    """Guard packed nibble order, E8M0 groups, and boundary masks."""
    torch.manual_seed(1201 + m + n + k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    packed_weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    weight_scale = torch.randint(
        110, 140, (n, k // 32), device="cuda", dtype=torch.uint8
    )
    reference = F.linear(x, _dequantize_reference(packed_weight, weight_scale))

    output = triton_mxfp4_small_m_linear(x, packed_weight, weight_scale)

    torch.testing.assert_close(output, reference, rtol=0, atol=0)


@pytest.mark.skipif(not _on_gfx1201(), reason="requires ROCm gfx1201")
@pytest.mark.parametrize("raw_scale", [0, 1, 127, 254, 255])
def test_small_m_matches_e8m0_edge_values(raw_scale: int):
    """Guard the finite extremes and reserved NaN encoding of E8M0."""
    x = torch.ones((1, 128), device="cuda", dtype=torch.bfloat16)
    packed_weight = torch.full((512, 64), 0x11, device="cuda", dtype=torch.uint8)
    weight_scale = torch.full((512, 4), raw_scale, device="cuda", dtype=torch.uint8)
    reference = F.linear(x, _dequantize_reference(packed_weight, weight_scale))

    output = triton_mxfp4_small_m_linear(x, packed_weight, weight_scale)

    torch.testing.assert_close(output, reference, rtol=0, atol=0, equal_nan=True)
