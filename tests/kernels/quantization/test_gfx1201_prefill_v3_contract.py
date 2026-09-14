# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone CPU checks for the inactive plan interfaces.

Run this file directly to avoid loading vLLM's GPU-dependent test fixtures.
These checks do not certify a GEMM, model quality, or performance.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "vllm/model_executor/kernels/linear/mxfp4/gfx1201_prefill_v3.py"
SPEC = importlib.util.spec_from_file_location("_prefill_v3_contract_test", MODULE)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


class TestInactivePrefillContract(unittest.TestCase):
    def setUp(self) -> None:
        self.call = contract.PrefillCall(
            arch="gfx1201",
            phase="prefill",
            owner="target",
            m=256,
            n=5120,
            k=6144,
            input_dtype="bfloat16",
        )

    def test_matching_profile_stays_disabled(self) -> None:
        self.assertIsNone(contract.structural_rejection(self.call))
        self.assertFalse(contract.can_use_gfx1201_prefill_v3(self.call))

    def test_profile_rejections(self) -> None:
        cases = (
            {"arch": "gfx950"},
            {"phase": "decode"},
            {"phase": "unknown"},
            {"owner": "drafter"},
            {"owner": "vision"},
            {"m": 4},
            {"n": 96},
            {"k": 0},
            {"k": 65},
            {"input_dtype": "float16"},
            {"has_bias": True},
            {"last_dim_contiguous": False},
            {"tensor_parallel_size": 2},
            {"graph_capture": True},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                call = replace(self.call, **changes)
                self.assertIsNotNone(contract.structural_rejection(call))
                self.assertFalse(contract.can_use_gfx1201_prefill_v3(call))

    def test_lossless_layout_is_not_numerical_folding(self) -> None:
        self.assertNotEqual(
            contract.WeightNumerics.GROUP_SCALED,
            contract.WeightNumerics.FOLDED_E4M3,
        )
        self.assertNotEqual(
            contract.WeightLayout.CHECKPOINT, contract.WeightLayout.FRAGMENT_V1
        )

    def test_prepare_fails_without_touching_inputs(self) -> None:
        with self.assertRaises(NotImplementedError):
            contract.prepare_weights(
                packed=None,
                scales=None,
                layout=contract.WeightLayout.CHECKPOINT,
                numerics=contract.WeightNumerics.GROUP_SCALED,
            )

    def test_workspace_fails(self) -> None:
        with self.assertRaises(NotImplementedError):
            contract.allocate_workspace(
                row_capacity=256, k_capacity=6144, n_capacity=5120, device=None
            )

    def test_quantization_fails(self) -> None:
        with self.assertRaises(NotImplementedError):
            contract.quantize_activation(x=None, output_bytes=None, output_scales=None)

    def test_raw_launch_fails(self) -> None:
        with self.assertRaises(NotImplementedError):
            contract.launch_raw_fp8(
                a_bytes=None, b_bytes=None, output=None, variant="H1"
            )

    def test_packed_launch_fails(self) -> None:
        with self.assertRaises(NotImplementedError):
            contract.launch_prefill(
                x=None, weights=None, workspace=None, call=self.call
            )


if __name__ == "__main__":
    unittest.main()
