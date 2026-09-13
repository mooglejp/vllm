# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.mxfp4.gfx1201_w4a8 import (
    FP8_E4M3_MAX,
    dequantize_fp8_activation_reference,
    dequantize_mxfp4_weight_reference,
    gfx1201_w4a8_linear_reference,
    is_gfx1201_w4a8_candidate,
    is_gfx1201_w4a8_prefill_candidate,
    quantize_activation_fp8_reference,
)


def _pack_codewords(codewords: torch.Tensor) -> torch.Tensor:
    return (codewords[:, 0::2] | (codewords[:, 1::2] << 4)).to(torch.uint8)


def test_activation_reference_uses_row_scales_and_finite_zero_rows():
    x = torch.tensor(
        [[2.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    quantized, scale = quantize_activation_fp8_reference(x)

    assert quantized.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        scale,
        torch.tensor([2.0 / FP8_E4M3_MAX, 1.0], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    expected_quantized = torch.tensor(
        [[448.0, -224.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        quantized.to(torch.float32), expected_quantized, rtol=0, atol=0
    )
    dequantized = dequantize_fp8_activation_reference(quantized, scale)
    torch.testing.assert_close(dequantized, x.to(torch.float32), rtol=0, atol=0)
    assert torch.isfinite(dequantized).all()


def test_activation_reference_clamps_to_finite_fp8_range():
    x = torch.tensor([[1000.0, -1000.0]], dtype=torch.bfloat16)
    quantized, scale = quantize_activation_fp8_reference(x)

    assert quantized.to(torch.float32).abs().max().item() == FP8_E4M3_MAX
    dequantized = dequantize_fp8_activation_reference(quantized, scale)
    torch.testing.assert_close(dequantized, x.to(torch.float32), rtol=0, atol=1)


def test_mxfp4_reference_decodes_nibbles_and_e8m0_edges():
    codewords = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * 2],
        dtype=torch.uint8,
    )
    packed = _pack_codewords(codewords)
    scales = torch.tensor([[127]], dtype=torch.uint8)
    decoded = dequantize_mxfp4_weight_reference(packed, scales)
    expected = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
        * 2,
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoded[0], expected, rtol=0, atol=0)

    half_decoded = dequantize_mxfp4_weight_reference(
        packed, torch.tensor([[126]], dtype=torch.uint8)
    )
    torch.testing.assert_close(half_decoded[0], expected * 0.5, rtol=0, atol=0)

    nan_decoded = dequantize_mxfp4_weight_reference(
        packed, torch.tensor([[255]], dtype=torch.uint8)
    )
    assert torch.isnan(nan_decoded).all()


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_w4a8_reference_matches_its_fp32_linear_contract(m: int):
    torch.manual_seed(1201 + m)
    n, k = 5, 64
    x = torch.randn((m, k), dtype=torch.bfloat16)
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    scales = torch.randint(120, 132, (n, k // 32), dtype=torch.uint8)

    quantized_x, activation_scale = quantize_activation_fp8_reference(x)
    expected = F.linear(
        dequantize_fp8_activation_reference(quantized_x, activation_scale),
        dequantize_mxfp4_weight_reference(packed, scales),
    ).to(torch.bfloat16)
    actual = gfx1201_w4a8_linear_reference(x, packed, scales)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mxfp4_reference_rejects_invalid_geometry():
    invalid_k = torch.zeros((2, 24), dtype=torch.uint8)
    invalid_scale = torch.zeros((2, 1), dtype=torch.uint8)
    with pytest.raises(ValueError, match="divisible"):
        dequantize_mxfp4_weight_reference(invalid_k, invalid_scale)

    packed = torch.zeros((2, 16), dtype=torch.uint8)
    wrong_scale = torch.zeros((2, 2), dtype=torch.uint8)
    with pytest.raises(ValueError, match="shape"):
        dequantize_mxfp4_weight_reference(packed, wrong_scale)


def test_w4a8_candidate_accepts_only_decode_shapes():
    x = torch.zeros((1, 32), dtype=torch.bfloat16)
    assert is_gfx1201_w4a8_candidate(x=x, bias=None)
    assert not is_gfx1201_w4a8_candidate(
        x=torch.empty((1, 0), dtype=torch.bfloat16), bias=None
    )
    assert not is_gfx1201_w4a8_candidate(
        x=torch.zeros((32, 1), dtype=torch.bfloat16).expand(32, 32), bias=None
    )
    assert not is_gfx1201_w4a8_candidate(
        x=torch.zeros((5, 32), dtype=torch.bfloat16), bias=None
    )
    assert not is_gfx1201_w4a8_candidate(
        x=torch.zeros((1, 24), dtype=torch.bfloat16), bias=None
    )
    assert not is_gfx1201_w4a8_candidate(
        x=torch.zeros((1, 32), dtype=torch.float16), bias=None
    )
    assert not is_gfx1201_w4a8_candidate(
        x=torch.zeros((1, 32), dtype=torch.bfloat16),
        bias=torch.zeros(1, dtype=torch.bfloat16),
    )


def test_w4a8_prefill_candidate_requires_large_m_and_group32_layout():
    x = torch.zeros((128, 64), dtype=torch.bfloat16)
    packed = torch.zeros((16, 32), dtype=torch.uint8)
    scale = torch.zeros((16, 2), dtype=torch.uint8)
    assert is_gfx1201_w4a8_prefill_candidate(
        x=x, packed_weight=packed, weight_scale=scale
    )
    assert not is_gfx1201_w4a8_prefill_candidate(
        x=x[:127], packed_weight=packed, weight_scale=scale
    )
    assert not is_gfx1201_w4a8_prefill_candidate(
        x=torch.zeros((128, 48), dtype=torch.bfloat16),
        packed_weight=torch.zeros((16, 24), dtype=torch.uint8),
        weight_scale=torch.zeros((16, 1), dtype=torch.uint8),
    )
    assert not is_gfx1201_w4a8_prefill_candidate(
        x=x,
        packed_weight=packed,
        weight_scale=torch.zeros((16, 1), dtype=torch.uint8),
    )
    assert not is_gfx1201_w4a8_prefill_candidate(
        x=x,
        packed_weight=torch.zeros((0, 32), dtype=torch.uint8),
        weight_scale=torch.zeros((0, 2), dtype=torch.uint8),
    )


def test_w4a8_prefill_candidate_rejects_noncontiguous_final_dimension():
    x = torch.zeros((128, 128), dtype=torch.bfloat16)[:, ::2]
    packed = torch.zeros((16, 32), dtype=torch.uint8)
    scale = torch.zeros((16, 2), dtype=torch.uint8)
    assert x.shape == (128, 64)
    assert x.stride(-1) == 2
    assert not is_gfx1201_w4a8_prefill_candidate(
        x=x, packed_weight=packed, weight_scale=scale
    )
