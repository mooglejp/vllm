# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuant KV-cache quantization.

Run: .venv/bin/python -m pytest tests/quantization/test_turboquant.py -v
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
    solve_lloyd_max,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2

# ============================================================================
# Helpers
# ============================================================================

ALL_PRESETS = list(TQ_PRESETS.keys())


def _assert_strictly_sorted(seq, name="sequence"):
    for i in range(len(seq) - 1):
        assert seq[i] < seq[i + 1], f"{name} not sorted at index {i}"


def _is_power_of_2(n: int) -> bool:
    return n > 0 and next_power_of_2(n) == n


# Expected concrete values for each preset at head_dim=128.
# fmt: off
PRESET_EXPECTED = {
    "turboquant_k8v4": dict(
        key_fp8=True,  key_quant_bits=8,
        key_mse_bits=0, value_quant_bits=4,
        mse_bits=4, n_centroids=16, centroid_bits=4,
        norm_correction=False,
        key_packed_size=128, value_packed_size=68,
        slot_size=196, slot_size_aligned=196,
    ),
    "turboquant_4bit_nc": dict(
        key_fp8=False, key_quant_bits=4,
        key_mse_bits=4, value_quant_bits=4,
        mse_bits=4, n_centroids=16, centroid_bits=4,
        norm_correction=True,
        key_packed_size=66, value_packed_size=68,
        slot_size=134, slot_size_aligned=134,
    ),
    "turboquant_k3v4_nc": dict(
        key_fp8=False, key_quant_bits=3,
        key_mse_bits=3, value_quant_bits=4,
        mse_bits=3, n_centroids=8, centroid_bits=3,
        norm_correction=True,
        key_packed_size=50, value_packed_size=68,
        slot_size=118, slot_size_aligned=118,
    ),
    "turboquant_3bit_nc": dict(
        key_fp8=False, key_quant_bits=3,
        key_mse_bits=3, value_quant_bits=3,
        mse_bits=3, n_centroids=8, centroid_bits=3,
        norm_correction=True,
        key_packed_size=50, value_packed_size=52,
        slot_size=102, slot_size_aligned=102,
    ),
}
# fmt: on


# ============================================================================
# Config tests (CPU-only, no dependencies beyond config.py)
# ============================================================================


class TestTurboQuantConfig:
    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_preset_parses(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert isinstance(cfg, TurboQuantConfig)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown TurboQuant"):
            TurboQuantConfig.from_cache_dtype("turboquant_invalid", head_dim=128)

    # ---- Per-preset concrete value checks (table-driven) ----

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_key_mode(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.key_fp8 is exp["key_fp8"]
        assert cfg.key_quant_bits == exp["key_quant_bits"]
        assert cfg.key_mse_bits == exp["key_mse_bits"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_value_mode(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.value_quant_bits == exp["value_quant_bits"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_bits_and_centroids(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.mse_bits == exp["mse_bits"]
        assert cfg.n_centroids == exp["n_centroids"]
        assert cfg.centroid_bits == exp["centroid_bits"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_norm_correction(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.norm_correction is PRESET_EXPECTED[preset]["norm_correction"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_packed_sizes(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.key_packed_size == exp["key_packed_size"]
        assert cfg.value_packed_size == exp["value_packed_size"]
        assert cfg.slot_size == exp["slot_size"]
        assert cfg.slot_size_aligned == exp["slot_size_aligned"]

    # ---- Cross-preset structural invariants ----

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_slot_equals_key_plus_value(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.slot_size == cfg.key_packed_size + cfg.value_packed_size

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_padded_slot_is_even(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.slot_size_aligned >= cfg.slot_size
        assert cfg.slot_size_aligned % 2 == 0, (
            f"slot_size_aligned={cfg.slot_size_aligned} is not even"
        )

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_key_value_packed_sizes_positive(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.key_packed_size > 0
        assert cfg.value_packed_size > 0

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_n_centroids_is_2_to_mse_bits(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.n_centroids == 2**cfg.mse_bits

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_centroid_bits_always_positive(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.centroid_bits > 0

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_mse_key_or_fp8_exclusive(self, preset):
        """Each preset is either FP8 keys or MSE keys, never both."""
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        if cfg.key_fp8:
            assert cfg.key_mse_bits == 0
            assert cfg.key_quant_bits == 8
        else:
            assert cfg.key_mse_bits > 0
            assert cfg.key_quant_bits in (3, 4)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
    def test_all_presets_all_head_dims(self, preset, head_dim):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=head_dim)
        assert cfg.head_dim == head_dim
        assert cfg.slot_size == cfg.key_packed_size + cfg.value_packed_size
        assert cfg.slot_size_aligned >= cfg.slot_size
        assert cfg.slot_size_aligned % 2 == 0

    # ---- Boundary skip layers ----

    @staticmethod
    def _dense_model_config(num_layers):
        from types import SimpleNamespace

        return SimpleNamespace(
            is_hybrid=False,
            hf_text_config=SimpleNamespace(num_hidden_layers=num_layers),
        )

    def test_boundary_skip_layers_basic(self):
        mc = self._dense_model_config(32)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc)
        assert layers == ["0", "1", "30", "31"]

    def test_boundary_skip_layers_zero(self):
        mc = self._dense_model_config(32)
        assert TurboQuantConfig.get_boundary_skip_layers(mc, 0) == []

    def test_boundary_skip_layers_small_model(self):
        mc = self._dense_model_config(4)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc)
        assert layers == ["0", "1", "2", "3"]

    def test_boundary_skip_layers_cap_at_half(self):
        mc = self._dense_model_config(8)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc, 10)
        assert len(layers) == 8


class TestHybridAttentionIndices:
    """Regression tests for boundary protection on hybrid models.

    Hybrid models (attention + Mamba / linear-attention) identify KV-carrying
    layers via layer_types / layers_block_type / attn_type_list. The helper
    must return the *global* layer indices of the full-attention layers so
    that kv_cache_dtype_skip_layers matches what extract_layer_index(prefix)
    reports on the Attention layers at runtime.
    """

    @staticmethod
    def _fake_model_config(text_cfg=None, hf_cfg=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            hf_text_config=text_cfg if text_cfg is not None else SimpleNamespace(),
            hf_config=hf_cfg if hf_cfg is not None else SimpleNamespace(),
        )

    def test_layer_types_full_attention(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        cfg = type("C", (), {})()
        cfg.layer_types = [
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
            "full_attention",
        ]
        mc = self._fake_model_config(text_cfg=cfg)
        assert _get_full_attention_layer_indices(mc) == [2, 4, 5]

    def test_layers_block_type_jamba(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        cfg = type("C", (), {})()
        cfg.layers_block_type = ["mamba", "attention", "mamba", "attention"]
        mc = self._fake_model_config(text_cfg=cfg)
        assert _get_full_attention_layer_indices(mc) == [1, 3]

    def test_attn_type_list_minimax(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        hf = type("C", (), {})()
        hf.attn_type_list = [0, 1, 0, 1, 1]
        mc = self._fake_model_config(hf_cfg=hf)
        assert _get_full_attention_layer_indices(mc) == [1, 3, 4]

    def test_no_hybrid_hints_returns_empty(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        mc = self._fake_model_config()
        assert _get_full_attention_layer_indices(mc) == []


class TestGfx1201TargetProfile:
    @staticmethod
    def _profile() -> dict:
        return {
            "rocm_arch": "gfx1201",
            "head_size": 256,
            "num_kv_groups": 6,
            "key_fp8": True,
            "value_quant_bits": 4,
            "block_size": 16,
            "has_sinks": False,
            "sliding_window": None,
        }

    def test_accepts_exact_profile(self):
        from vllm.v1.attention.backends.turboquant_gfx1201_k8v4_mtp import (
            is_target_profile,
        )

        assert is_target_profile(**self._profile())

    def test_host_mirror_enforces_single_token_contract(self):
        from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode_gfx1201_k8v4 import (  # noqa: E501
            _validate_single_token_inputs,
        )

        with pytest.raises(ValueError, match="single-token"):
            _validate_single_token_inputs(
                query=torch.empty(1, 6, 256, dtype=torch.float16),
                kv_cache=torch.empty(1, 16, 1, 388, dtype=torch.uint8),
                block_table=torch.zeros(1, 1, dtype=torch.int32),
                seq_lens=torch.ones(1, dtype=torch.int32),
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                max_num_kv_splits=1,
                query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
            )

    @pytest.mark.parametrize("num_tokens", [6, 8])
    def test_host_mirror_accepts_ragged_multi_token_contract(self, num_tokens):
        from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode_gfx1201_k8v4 import (  # noqa: E501
            _validate_multi_token_inputs,
        )

        result = _validate_multi_token_inputs(
            query=torch.empty(num_tokens, 12, 256, dtype=torch.float16),
            kv_cache=torch.empty(4, 16, 2, 388, dtype=torch.uint8),
            block_table=torch.zeros(3, 3, dtype=torch.int32),
            seq_lens=torch.tensor([20, 31, 33], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2, 3, 6], dtype=torch.int32),
            max_num_kv_splits=4,
            query_start_loc_cpu=torch.tensor([0, 2, 3, 6], dtype=torch.int32),
        )

        assert result == (3, 12, 16)

    def test_host_mirror_rejects_multi_token_query_count_mismatch(self):
        from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode_gfx1201_k8v4 import (  # noqa: E501
            _validate_multi_token_inputs,
        )

        with pytest.raises(ValueError, match="end within the query buffer"):
            _validate_multi_token_inputs(
                query=torch.empty(6, 12, 256, dtype=torch.float16),
                kv_cache=torch.empty(4, 16, 2, 388, dtype=torch.uint8),
                block_table=torch.zeros(2, 3, dtype=torch.int32),
                seq_lens=torch.tensor([20, 31], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 2, 7], dtype=torch.int32),
                max_num_kv_splits=4,
                query_start_loc_cpu=torch.tensor([0, 2, 7], dtype=torch.int32),
            )

    def test_backend_opts_out_of_adaptive_query_boundaries(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )

        assert not TurboQuantAttentionBackend.supports_device_cpu_query_lens_mismatch()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rocm_arch", "gfx1200"),
            ("head_size", 128),
            ("num_kv_groups", 8),
            ("key_fp8", False),
            ("value_quant_bits", 3),
            ("block_size", 64),
            ("has_sinks", True),
            ("sliding_window", 128),
        ],
    )
    def test_rejects_unsupported_profile_field(self, field, value):
        from vllm.v1.attention.backends.turboquant_gfx1201_k8v4_mtp import (
            is_target_profile,
        )

        profile = self._profile()
        profile[field] = value
        assert not is_target_profile(**profile)

    def test_backend_gate_requires_opt_in_and_exact_profile(self, monkeypatch):
        from vllm.v1.attention.backends import turboquant_attn

        profile = self._profile()
        monkeypatch.setattr(
            turboquant_attn,
            "_runtime_rocm_arch",
            lambda: profile["rocm_arch"],
        )
        monkeypatch.setattr(
            turboquant_attn.envs,
            "VLLM_TQ_GFX1201_K8V4",
            False,
        )
        assert not turboquant_attn._should_use_gfx1201_fast_path(
            head_size=profile["head_size"],
            num_kv_groups=profile["num_kv_groups"],
            key_fp8=profile["key_fp8"],
            value_quant_bits=profile["value_quant_bits"],
            block_size=profile["block_size"],
            has_sinks=profile["has_sinks"],
            sliding_window=profile["sliding_window"],
        )

        monkeypatch.setattr(
            turboquant_attn.envs,
            "VLLM_TQ_GFX1201_K8V4",
            True,
        )
        assert turboquant_attn._should_use_gfx1201_fast_path(
            head_size=profile["head_size"],
            num_kv_groups=profile["num_kv_groups"],
            key_fp8=profile["key_fp8"],
            value_quant_bits=profile["value_quant_bits"],
            block_size=profile["block_size"],
            has_sinks=profile["has_sinks"],
            sliding_window=profile["sliding_window"],
        )


class TestTurboQuantKVCacheSpec:
    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_kv_cache_spec_sets_kv_quant_mode(self, preset):
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        layer = SimpleNamespace(
            attn_type="decoder",
            kv_cache_dtype=preset,
            kv_cache_torch_dtype=torch.uint8,
            head_size=128,
            head_size_v=128,
            num_kv_heads=4,
            sliding_window=None,
            get_attn_backend=lambda: TurboQuantAttentionBackend,
        )
        vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=32))

        # The layer builds an unpacked spec; the worker's spec-collection
        # loop applies TQ slot packing via the backend's customize_spec hook.
        spec = Attention.get_kv_cache_spec(layer, vllm_config)
        assert isinstance(spec, FullAttentionSpec)
        assert spec.kv_quant_mode.is_turboquant
        assert spec.state_content_bytes is None

        spec = TurboQuantAttentionBackend.customize_spec(spec)
        expected_slot = TurboQuantConfig.from_cache_dtype(preset, 128).slot_size_aligned
        assert spec.state_content_bytes == expected_slot


class TestTurboQuantWorkspaceReservation:
    @staticmethod
    def _fake_vllm_config(
        *,
        max_num_seqs: int = 16,
        max_num_batched_tokens: int = 4096,
        enable_chunked_prefill: bool = True,
        max_model_len: int = 8192,
        dtype: torch.dtype = torch.float16,
        max_num_kv_splits: int = 4,
        num_attention_heads: int = 8,
        speculative_config=None,
        capture_sizes=None,
    ):
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=enable_chunked_prefill,
            ),
            model_config=SimpleNamespace(
                max_model_len=max_model_len,
                dtype=dtype,
                get_num_attention_heads=lambda parallel_config: num_attention_heads,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=2,
                decode_context_parallel_size=1,
            ),
            attention_config=SimpleNamespace(
                tq_max_kv_splits_for_cuda_graph=max_num_kv_splits
            ),
            speculative_config=speculative_config,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=capture_sizes),
            cache_config=SimpleNamespace(block_size=16),
        )

    @staticmethod
    def _fake_kv_cache_spec():
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        return FullAttentionSpec(
            block_size=32,
            num_kv_heads=4,
            head_size=128,
            head_size_v=128,
            dtype=torch.uint8,
            state_content_bytes=102,
        )

    @pytest.mark.parametrize(
        ("token_budget", "capture_sizes", "expected_tokens"),
        [(4096, None, 15), (8, [8], 8), (32, [16], 16)],
    )
    def test_target_spec_decode_reserves_by_query_token(
        self, monkeypatch, token_budget, capture_sizes, expected_tokens
    ):
        from vllm.v1.attention.backends import turboquant_attn
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.TURBOQUANT_K8V4,
            state_content_bytes=388,
        )

        calls = []

        class FakeWorkspaceManager:
            def get_simultaneous(self, *shapes_and_dtypes):
                calls.append(shapes_and_dtypes)

        monkeypatch.setattr(
            turboquant_attn,
            "current_workspace_manager",
            lambda: FakeWorkspaceManager(),
        )
        monkeypatch.setattr(
            turboquant_attn,
            "is_workspace_manager_initialized",
            lambda: True,
        )
        monkeypatch.setattr(turboquant_attn, "_runtime_rocm_arch", lambda: "gfx1201")
        monkeypatch.setattr(turboquant_attn.envs, "VLLM_TQ_GFX1201_K8V4", True)

        builder = turboquant_attn.TurboQuantMetadataBuilder(
            kv_cache_spec=spec,
            layer_names=["layers.0.self_attn.attn"],
            vllm_config=self._fake_vllm_config(
                max_num_seqs=3,
                max_num_batched_tokens=token_budget,
                capture_sizes=capture_sizes,
                num_attention_heads=24,
                speculative_config=SimpleNamespace(
                    num_speculative_tokens=2,
                    parallel_drafting=True,
                ),
            ),
            device=torch.device("cuda"),
        )

        assert builder.reorder_batch_threshold == 5
        assert calls[0] == (
            ((expected_tokens, 24, 4, 257), torch.float32),
            ((expected_tokens, 24, 256), torch.float16),
            ((expected_tokens, 24), torch.float32),
        )

    def test_metadata_builder_reserves_decode_and_continuation_prefill_workspace(
        self, monkeypatch
    ):
        from vllm.v1.attention.backends import turboquant_attn

        calls = []

        class FakeWorkspaceManager:
            def get_simultaneous(self, *shapes_and_dtypes):
                calls.append(shapes_and_dtypes)

        monkeypatch.setattr(
            turboquant_attn,
            "current_workspace_manager",
            lambda: FakeWorkspaceManager(),
        )
        monkeypatch.setattr(
            turboquant_attn,
            "is_workspace_manager_initialized",
            lambda: True,
        )

        turboquant_attn.TurboQuantMetadataBuilder(
            kv_cache_spec=self._fake_kv_cache_spec(),
            layer_names=["layers.0.self_attn.attn"],
            vllm_config=self._fake_vllm_config(),
            device=torch.device("cuda"),
        )

        assert calls == [
            (
                ((16, 8, 4, 129), torch.float32),
                ((16, 8, 128), torch.float16),
                ((16, 8), torch.float32),
            ),
            (
                ((1, 4, 8192, 128), torch.float16),
                ((1, 4, 8192, 128), torch.float16),
            ),
        ]

    def test_metadata_builder_skips_continuation_prefill_when_disabled(
        self, monkeypatch
    ):
        from vllm.v1.attention.backends import turboquant_attn

        calls = []

        class FakeWorkspaceManager:
            def get_simultaneous(self, *shapes_and_dtypes):
                calls.append(shapes_and_dtypes)

        monkeypatch.setattr(
            turboquant_attn,
            "current_workspace_manager",
            lambda: FakeWorkspaceManager(),
        )
        monkeypatch.setattr(
            turboquant_attn,
            "is_workspace_manager_initialized",
            lambda: True,
        )

        turboquant_attn.TurboQuantMetadataBuilder(
            kv_cache_spec=self._fake_kv_cache_spec(),
            layer_names=["layers.0.self_attn.attn"],
            vllm_config=self._fake_vllm_config(enable_chunked_prefill=False),
            device=torch.device("cuda"),
        )

        assert calls == [
            (
                ((16, 8, 4, 129), torch.float32),
                ((16, 8, 128), torch.float16),
                ((16, 8), torch.float32),
            )
        ]


# ============================================================================
# Centroids tests (CPU-only)
# ============================================================================


class TestCentroids:
    @pytest.mark.parametrize("bits,expected_n", [(2, 4), (3, 8), (4, 16)])
    def test_centroids_shape(self, bits, expected_n):
        c = get_centroids(128, bits)
        assert c.shape == (expected_n,)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_sorted(self, bits):
        _assert_strictly_sorted(get_centroids(128, bits), "centroids")

    def test_centroids_cached(self):
        c1 = get_centroids(128, 3)
        c2 = get_centroids(128, 3)
        assert c1 is c2, "get_centroids should return cached object"

    def test_centroids_different_dims_not_identical(self):
        c64 = get_centroids(64, 3)
        c128 = get_centroids(128, 3)
        assert not torch.equal(c64, c128)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_symmetric_around_zero(self, bits):
        """N(0, 1/d) is symmetric, so centroids should be ~symmetric."""
        c = get_centroids(128, bits)
        assert abs(c.mean().item()) < 0.01, "Centroids not centered near 0"
        assert abs(c[0].item() + c[-1].item()) < 0.01

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_within_4sigma(self, bits):
        """All centroids should be within ~4 sigma of N(0, 1/d)."""
        sigma = math.sqrt(1.0 / 128)
        c = get_centroids(128, bits)
        for i, val in enumerate(c):
            assert abs(val.item()) < 4 * sigma, (
                f"Centroid {i}={val:.6f} outside 4*sigma={4 * sigma:.6f}"
            )


class TestLloydMax:
    @pytest.mark.parametrize("bits,expected_n", [(2, 4), (3, 8), (4, 16)])
    def test_solve_shapes(self, bits, expected_n):
        centroids, boundaries = solve_lloyd_max(128, bits)
        assert centroids.shape == (expected_n,)
        assert boundaries.shape == (expected_n - 1,)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_sorted(self, bits):
        centroids, _ = solve_lloyd_max(128, bits)
        _assert_strictly_sorted(centroids, "centroids")

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_boundaries_sorted(self, bits):
        _, boundaries = solve_lloyd_max(128, bits)
        _assert_strictly_sorted(boundaries, "boundaries")

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_boundaries_between_centroids(self, bits):
        """Each boundary must lie between its adjacent centroids."""
        centroids, boundaries = solve_lloyd_max(128, bits)
        for i in range(len(boundaries)):
            assert centroids[i] < boundaries[i] < centroids[i + 1], (
                f"Boundary {i}={boundaries[i]:.6f} not between "
                f"c[{i}]={centroids[i]:.6f} and c[{i + 1}]={centroids[i + 1]:.6f}"
            )

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_boundaries_are_midpoints(self, bits):
        """Lloyd-Max boundaries are midpoints of adjacent centroids."""
        centroids, boundaries = solve_lloyd_max(128, bits)
        for i in range(len(boundaries)):
            expected = (centroids[i] + centroids[i + 1]) / 2.0
            assert abs(boundaries[i].item() - expected.item()) < 1e-6

    def test_solve_deterministic(self):
        c1, b1 = solve_lloyd_max(128, 3)
        c2, b2 = solve_lloyd_max(128, 3)
        assert torch.equal(c1, c2)
        assert torch.equal(b1, b2)

    def test_solve_dtype_float32(self):
        centroids, boundaries = solve_lloyd_max(128, 3)
        assert centroids.dtype == torch.float32
        assert boundaries.dtype == torch.float32

    @pytest.mark.parametrize("bits", [3, 4])
    def test_centroids_match_scipy_reference(self, bits):
        """Verify _trapz(n=200) centroids match scipy.integrate.quad reference.

        This ensures our scipy-free trapezoid integration doesn't silently
        drift from the published Lloyd-Max quality.
        """
        pytest.importorskip("scipy")
        from scipy.integrate import quad

        d = 128
        sigma2 = 1.0 / d
        sigma = math.sqrt(sigma2)

        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(
                -x * x / (2 * sigma2)
            )

        n_levels = 2**bits
        lo, hi = -3.5 * sigma, 3.5 * sigma
        ref_centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
        for _ in range(200):
            boundaries = [
                (ref_centroids[i] + ref_centroids[i + 1]) / 2.0
                for i in range(n_levels - 1)
            ]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_centroids = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = quad(lambda x: x * pdf(x), a, b)
                den, _ = quad(pdf, a, b)
                new_centroids.append(num / den if den > 1e-15 else ref_centroids[i])
            if (
                max(abs(new_centroids[i] - ref_centroids[i]) for i in range(n_levels))
                < 1e-10
            ):
                break
            ref_centroids = new_centroids

        # Compare our _trapz centroids against scipy reference
        our_centroids, _ = solve_lloyd_max(d, bits)
        ref_t = torch.tensor(ref_centroids, dtype=torch.float32)
        max_err = (our_centroids - ref_t).abs().max().item()
        # _trapz(n=200) has ~O(h^2) error vs adaptive quad; 1e-3 is tight
        # enough to catch regression while allowing trapezoid approximation.
        assert max_err < 1e-3, (
            f"d={d}, bits={bits}: max centroid error vs scipy = {max_err:.2e}"
        )


# ============================================================================
# Rotation matrix tests (GPU required)
# ============================================================================

GPGPU_AVAILABLE = torch.cuda.is_available() or torch.xpu.is_available()
DEVICE_TYPE = current_platform.device_type


def _on_gfx1201() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import _GCN_ARCH

    return _GCN_ARCH == "gfx1201"


def generate_rotation_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Haar-distributed random orthogonal matrix via QR (test/benchmark only)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    # torch.linalg.qr on CPU requires LAPACK, which some torch wheels
    # (ROCm) ship without. Run QR on accelerator instead
    qr_device = "cuda" if torch.cuda.is_available() else "cpu"
    Q, R = torch.linalg.qr(G.to(qr_device))
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestRotationMatrix:
    """Tests for the QR-based rotation (standalone benchmarks only)."""

    @pytest.mark.parametrize("dim", [64, 96, 128, 256])
    def test_rotation_matrix_shape_and_orthogonal(self, dim):
        Pi = generate_rotation_matrix(dim, seed=42, device=DEVICE_TYPE)
        assert Pi.shape == (dim, dim)
        eye = Pi @ Pi.T
        assert torch.allclose(eye, torch.eye(dim, device=DEVICE_TYPE), atol=1e-5), (
            f"Pi not orthogonal for dim={dim}"
        )

    def test_rotation_matrix_deterministic(self):
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=42)
        assert torch.equal(Pi1, Pi2)

    def test_rotation_matrix_different_seeds(self):
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=99)
        assert not torch.equal(Pi1, Pi2)

    def test_rotation_matrix_det_is_pm1(self):
        """Orthogonal matrix determinant must be +1 or -1."""
        Pi = generate_rotation_matrix(128, seed=42, device=DEVICE_TYPE)
        det = torch.linalg.det(Pi)
        assert abs(abs(det.item()) - 1.0) < 1e-4


# ============================================================================
# Hadamard rotation tests (serving path: _build_hadamard)
# ============================================================================


def _build_hadamard(d: int, device: str = "cpu") -> torch.Tensor:
    """Reproduce the serving-path Hadamard construction."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device))


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestHadamardRotation:
    """Tests for the Hadamard rotation used in serving."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_hadamard_orthonormal(self, dim):
        """H must be orthonormal: H @ H^T = I."""
        H = _build_hadamard(dim, DEVICE_TYPE)
        eye = H @ H.T
        assert torch.allclose(eye, torch.eye(dim, device=DEVICE_TYPE), atol=1e-5), (
            f"Hadamard not orthonormal for dim={dim}"
        )

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_hadamard_symmetric(self, dim):
        """Sylvester Hadamard must be symmetric: H = H^T."""
        H = _build_hadamard(dim, DEVICE_TYPE)
        assert torch.allclose(H, H.T, atol=1e-6), (
            f"Hadamard not symmetric for dim={dim}"
        )


# ============================================================================
# Store → Decode round-trip test (GPU + Triton required)
# ============================================================================


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestStoreDecodeRoundTrip:
    """End-to-end: store KV into TQ cache, decode, compare vs fp16 ref."""

    @pytest.mark.parametrize(
        "preset",
        ["turboquant_k8v4", "turboquant_4bit_nc"],
    )
    def test_single_token_roundtrip(self, preset):
        """Store 1 token, decode with query=key, check attention output.

        For a single token with query=key, attention output should equal
        the value (softmax over single key = 1.0). Quantization error
        means we check cosine similarity rather than exact equality.
        """
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            solve_lloyd_max,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        D = 128
        Hk = 4  # num_kv_heads
        Hq = 4  # num_q_heads (no GQA for simplicity)
        B = 1  # single token
        block_size = 16
        num_blocks = 1

        device = torch.device(DEVICE_TYPE)

        # Pure Hadamard rotation (symmetric: H = H^T, so Pi = PiT = H)
        H = _build_hadamard(D, DEVICE_TYPE)
        PiT = H
        Pi = H

        # Generate centroids
        centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
        centroids = centroids.float().to(device)
        c_sorted, _ = centroids.sort()
        midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

        # Random K, V
        torch.manual_seed(123)
        key = torch.randn(B, Hk, D, device=device, dtype=torch.float16)
        value = torch.randn(B, Hk, D, device=device, dtype=torch.float16)

        # Allocate KV cache
        padded_slot = cfg.slot_size_aligned
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            Hk,
            padded_slot,
            device=device,
            dtype=torch.uint8,
        )
        slot_mapping = torch.tensor([0], device=device, dtype=torch.int32)

        # Store
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            PiT,
            midpoints,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )

        # Decode: use key as query so attention = softmax([1]) * V = V
        query = key.expand(B, Hq, D).contiguous().to(torch.float16)
        block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([1], device=device, dtype=torch.int32)

        output = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=1.0 / math.sqrt(D),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
            max_num_kv_splits=4,
        )

        # With single KV, output should approximate the stored value.
        # Check per-head cosine similarity > threshold.
        out_fp32 = output.float()
        val_fp32 = value.expand(B, Hq, D).float()
        for h in range(Hq):
            cos_sim = torch.nn.functional.cosine_similarity(
                out_fp32[0, h].unsqueeze(0),
                val_fp32[0, h].unsqueeze(0),
            ).item()
            # FP8 keys should be very accurate; MSE keys have more error
            threshold = 0.95 if cfg.key_fp8 else 0.85
            assert cos_sim > threshold, (
                f"Preset {preset} head {h}: cosine_sim={cos_sim:.4f} < {threshold}"
            )


@pytest.mark.skipif(
    not GPGPU_AVAILABLE or not _on_gfx1201(),
    reason="requires a ROCm gfx1201 device",
)
class TestGfx1201K8V4Decode:
    @pytest.mark.parametrize("block_size", [16, 32])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("context_len", [1024, 4096, 32768])
    def test_specialized_decode_matches_safe_soa(self, block_size, dtype, context_len):
        from vllm.config.attention import AttentionConfig
        from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode_gfx1201_k8v4 import (  # noqa: E501
            triton_turboquant_decode_gfx1201_k8v4,
        )
        from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_store import (
            triton_turboquant_store,
        )
        from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_unified_attention import (  # noqa: E501
            triton_turboquant_decode_attention_soa,
        )

        cfg = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", head_dim=256)
        device = torch.device(DEVICE_TYPE)
        batch = 2
        num_kv_heads = 8
        num_query_heads = num_kv_heads * 6
        seq_lengths = [context_len, context_len - 3]
        num_blocks_per_request = [
            math.ceil(seq_len / block_size) for seq_len in seq_lengths
        ]
        num_blocks = sum(num_blocks_per_request)
        max_num_blocks = max(num_blocks_per_request)
        block_table_cpu = torch.full((batch, max_num_blocks), -1, dtype=torch.int32)
        logical_block_ids = torch.arange(num_blocks, dtype=torch.int32)
        physical_block_ids = torch.cat(
            (logical_block_ids[::2], logical_block_ids[1::2])
        )
        block_offset = 0
        for request_idx, request_blocks in enumerate(num_blocks_per_request):
            block_table_cpu[request_idx, :request_blocks] = physical_block_ids[
                block_offset : block_offset + request_blocks
            ]
            block_offset += request_blocks
        assert block_offset == num_blocks
        block_table = block_table_cpu.to(device)
        seq_lens = torch.tensor(seq_lengths, device=device, dtype=torch.int32)

        torch.manual_seed(1201 + block_size + context_len)
        num_tokens = sum(seq_lengths)
        key = torch.randn(
            num_tokens,
            num_kv_heads,
            cfg.head_dim,
            device=device,
            dtype=dtype,
        )
        value = torch.randn_like(key)
        query = torch.randn(
            batch,
            num_query_heads,
            cfg.head_dim,
            device=device,
            dtype=dtype,
        )

        slot_mapping_cpu = []
        for req_idx, seq_len in enumerate(seq_lengths):
            for pos in range(seq_len):
                physical_block = block_table_cpu[req_idx, pos // block_size].item()
                slot_mapping_cpu.append(physical_block * block_size + pos % block_size)
        assert len(slot_mapping_cpu) == num_tokens
        slot_mapping = torch.tensor(slot_mapping_cpu, device=device, dtype=torch.int32)

        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            num_kv_heads,
            cfg.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        centroids = get_centroids(cfg.head_dim, cfg.centroid_bits).to(device)
        triton_turboquant_store(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            PiT=centroids,
            midpoints=centroids[:-1],
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )

        scale = 1.0 / math.sqrt(cfg.head_dim)
        qsl = torch.arange(batch + 1, device=device, dtype=torch.int32)
        qsl_cpu = torch.arange(batch + 1, dtype=torch.int32)
        max_num_kv_splits = AttentionConfig().tq_max_kv_splits_for_cuda_graph
        mid_o = torch.empty(
            batch,
            num_query_heads,
            max_num_kv_splits,
            cfg.head_dim + 1,
            device=device,
            dtype=torch.float32,
        )
        output_buffer = torch.empty_like(query)
        lse = torch.empty(batch, num_query_heads, device=device, dtype=torch.float32)
        fast_output = triton_turboquant_decode_gfx1201_k8v4(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            query_start_loc=qsl,
            query_start_loc_cpu=qsl_cpu,
            scale=scale,
            output=output_buffer,
            mid_o_buf=mid_o,
            lse_buf=lse,
            max_num_kv_splits=max_num_kv_splits,
            max_seq_len=context_len,
        )
        safe_output = triton_turboquant_decode_attention_soa(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=centroids,
            centroids=centroids,
            scale=scale,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            value_packed_size=cfg.value_packed_size,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            max_seq_len=context_len,
            max_num_kv_splits=max_num_kv_splits,
        )

        assert fast_output.data_ptr() == output_buffer.data_ptr()
        assert torch.isfinite(fast_output).all().item()
        torch.testing.assert_close(fast_output, safe_output, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("block_size", [16, 32])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        ("query_lengths", "seq_lengths"),
        [
            ([2, 4], [17, 33]),
            ([6, 1, 2], [20, 31, 33]),
            ([2, 0, 1], [17, 0, 33]),
        ],
    )
    def test_multi_token_decode_matches_safe_soa(
        self, block_size, dtype, query_lengths, seq_lengths
    ):
        from vllm.config.attention import AttentionConfig
        from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode_gfx1201_k8v4 import (  # noqa: E501
            triton_turboquant_decode_gfx1201_k8v4,
        )
        from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_store import (
            triton_turboquant_store,
        )
        from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_unified_attention import (  # noqa: E501
            triton_turboquant_unified_attention,
        )

        cfg = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", head_dim=256)
        device = torch.device(DEVICE_TYPE)
        num_kv_heads = 2
        num_query_heads = num_kv_heads * 6
        batch = len(seq_lengths)
        num_blocks_per_request = [
            math.ceil(seq_len / block_size) for seq_len in seq_lengths
        ]
        num_blocks = sum(num_blocks_per_request)
        max_num_blocks = max(num_blocks_per_request)
        block_table_cpu = torch.full((batch, max_num_blocks), -1, dtype=torch.int32)
        physical_block_ids = torch.roll(torch.arange(num_blocks, dtype=torch.int32), 1)
        block_offset = 0
        for request_idx, request_blocks in enumerate(num_blocks_per_request):
            block_table_cpu[request_idx, :request_blocks] = physical_block_ids[
                block_offset : block_offset + request_blocks
            ]
            block_offset += request_blocks
        block_table = block_table_cpu.to(device)
        seq_lens = torch.tensor(seq_lengths, device=device, dtype=torch.int32)

        torch.manual_seed(1201 + block_size + (dtype == torch.bfloat16))
        num_tokens = sum(seq_lengths)
        key = torch.randn(
            num_tokens, num_kv_heads, cfg.head_dim, device=device, dtype=dtype
        )
        value = torch.randn_like(key)
        num_query_tokens = sum(query_lengths)
        padded_tokens = num_query_tokens + 3
        query = torch.randn(
            padded_tokens,
            num_query_heads,
            cfg.head_dim,
            device=device,
            dtype=dtype,
        )

        slot_mapping_cpu = []
        for request_idx, seq_len in enumerate(seq_lengths):
            for position in range(seq_len):
                physical_block = block_table_cpu[
                    request_idx, position // block_size
                ].item()
                slot_mapping_cpu.append(
                    physical_block * block_size + position % block_size
                )
        slot_mapping = torch.tensor(slot_mapping_cpu, device=device, dtype=torch.int32)
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            num_kv_heads,
            cfg.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        centroids = get_centroids(cfg.head_dim, cfg.centroid_bits).to(device)
        triton_turboquant_store(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            PiT=centroids,
            midpoints=centroids[:-1],
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )

        scale = 1.0 / math.sqrt(cfg.head_dim)
        query_start_loc_cpu = [0]
        for query_len in query_lengths:
            query_start_loc_cpu.append(query_start_loc_cpu[-1] + query_len)
        qsl = torch.tensor(query_start_loc_cpu, device=device, dtype=torch.int32)
        qsl_cpu = torch.tensor(query_start_loc_cpu, dtype=torch.int32)
        max_num_kv_splits = AttentionConfig().tq_max_kv_splits_for_cuda_graph
        mid_o = torch.empty(
            padded_tokens,
            num_query_heads,
            max_num_kv_splits,
            cfg.head_dim + 1,
            device=device,
            dtype=torch.float32,
        )
        output_buffer = torch.full_like(query, 42)
        lse = torch.full(
            (padded_tokens, num_query_heads), 42, device=device, dtype=torch.float32
        )
        fast_output = triton_turboquant_decode_gfx1201_k8v4(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            query_start_loc=qsl,
            query_start_loc_cpu=qsl_cpu,
            scale=scale,
            output=output_buffer,
            mid_o_buf=mid_o,
            lse_buf=lse,
            max_num_kv_splits=max_num_kv_splits,
        )
        safe_output = triton_turboquant_unified_attention(
            query=query[:num_query_tokens],
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            query_start_loc=qsl,
            Pi=centroids,
            centroids=centroids,
            scale=scale,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            value_packed_size=cfg.value_packed_size,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            output=torch.empty_like(query),
            max_query_len=max(query_lengths),
            max_seq_len=max(seq_lengths),
            num_kv_splits=max_num_kv_splits,
            tile_size=16,
        )

        assert fast_output.data_ptr() == output_buffer.data_ptr()
        assert torch.isfinite(fast_output).all().item()
        torch.testing.assert_close(
            fast_output[:num_query_tokens],
            safe_output[:num_query_tokens],
            atol=2e-2,
            rtol=2e-2,
        )
        assert (output_buffer[num_query_tokens:] == 42).all()
        assert (lse[num_query_tokens:] == 42).all()

    @pytest.fixture
    def mtp_backend(self, monkeypatch):
        from vllm.config.attention import AttentionConfig
        from vllm.v1.attention.backends import turboquant_attn
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
        from vllm.v1.worker.workspace import WorkspaceManager

        device = torch.device(DEVICE_TYPE)
        config = TestTurboQuantWorkspaceReservation._fake_vllm_config(
            max_num_seqs=3,
            max_num_batched_tokens=32,
            max_model_len=64,
            enable_chunked_prefill=False,
            dtype=torch.bfloat16,
            num_attention_heads=12,
            max_num_kv_splits=AttentionConfig().tq_max_kv_splits_for_cuda_graph,
            speculative_config=SimpleNamespace(
                num_speculative_tokens=4, parallel_drafting=False
            ),
            capture_sizes=[16],
        )
        manager = WorkspaceManager(device)
        monkeypatch.setattr(turboquant_attn, "get_current_vllm_config", lambda: config)
        monkeypatch.setattr(
            turboquant_attn, "current_workspace_manager", lambda: manager
        )
        monkeypatch.setattr(
            turboquant_attn, "is_workspace_manager_initialized", lambda: True
        )
        monkeypatch.setattr(turboquant_attn, "is_flydsl_available", lambda: False)
        monkeypatch.setattr(turboquant_attn, "_HAS_FLASH_ATTN", False)
        monkeypatch.setattr(turboquant_attn.envs, "VLLM_TQ_GFX1201_K8V4", True)
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=256,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.TURBOQUANT_K8V4,
            state_content_bytes=388,
        )
        builder = turboquant_attn.TurboQuantMetadataBuilder(
            spec, ["attn"], config, device
        )
        impl = turboquant_attn.TurboQuantAttentionImpl(
            num_heads=12,
            head_size=256,
            scale=1 / 16,
            num_kv_heads=2,
            kv_cache_dtype="turboquant_k8v4",
        )
        assert impl._use_gfx1201_fast
        layer = SimpleNamespace()
        block_table_cpu = torch.tensor(
            [[8, 2, 5], [7, 1, 4], [6, 0, 3]], dtype=torch.int32
        )
        block_table = block_table_cpu.to(device)
        kv_cache = torch.zeros(
            9, 16, 2, 388, device=device, dtype=torch.uint8
        ).transpose(1, 2)
        torch.manual_seed(1201)
        key = torch.randn(3, 33, 2, 256, device=device, dtype=config.model_config.dtype)
        value = torch.randn_like(key)
        slots = torch.tensor(
            [
                int(block_table_cpu[r, p // 16]) * 16 + p % 16
                for r in range(3)
                for p in range(33)
            ],
            device=device,
            dtype=torch.int64,
        )
        impl.do_kv_cache_update(
            layer, key.flatten(0, 1), value.flatten(0, 1), kv_cache, slots
        )
        manager.lock()
        return SimpleNamespace(
            impl=impl,
            builder=builder,
            layer=layer,
            manager=manager,
            block_table=block_table,
            kv_cache=kv_cache,
            key=key,
            value=value,
        )

    @staticmethod
    def _mtp_common_metadata(data, query_lengths, seq_lengths, num_tokens):
        from vllm.v1.attention.backend import CommonAttentionMetadata

        qsl_cpu = torch.tensor([0, *query_lengths], dtype=torch.int32).cumsum(0).int()
        seq_lens_cpu = torch.tensor(seq_lengths, dtype=torch.int32)
        device = data.kv_cache.device
        return CommonAttentionMetadata(
            query_start_loc=qsl_cpu.to(device),
            query_start_loc_cpu=qsl_cpu,
            seq_lens=seq_lens_cpu.to(device),
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            num_reqs=3,
            num_actual_tokens=num_tokens,
            max_query_len=max(query_lengths),
            max_seq_len=max(seq_lengths),
            block_table_tensor=data.block_table,
            slot_mapping=torch.full(
                (num_tokens,), -1, device=device, dtype=torch.int64
            ),
        )

    @staticmethod
    def _mtp_reference(data, query, metadata):
        from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_unified_attention import (  # noqa: E501
            triton_turboquant_unified_attention,
        )

        cfg = data.impl.tq_config
        return triton_turboquant_unified_attention(
            query=query,
            kv_cache=data.kv_cache.transpose(1, 2),
            block_table=metadata.block_table_tensor,
            seq_lens=metadata.seq_lens,
            query_start_loc=metadata.query_start_loc,
            Pi=data.layer._tq_centroids,
            centroids=data.layer._tq_centroids,
            scale=data.impl.scale,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            value_packed_size=cfg.value_packed_size,
            key_fp8=True,
            output=torch.empty_like(query),
            max_query_len=metadata.max_query_len,
            max_seq_len=metadata.max_seq_len,
            num_kv_splits=data.impl.max_num_kv_splits,
            tile_size=16,
        )

    def test_mixed_mtp_forward_matches_decode_and_prefill_references(self, mtp_backend):
        """Packed decode offsets must not shift the subsequent raw-KV prefill."""
        data = mtp_backend
        common = self._mtp_common_metadata(data, [4, 2, 8], [33, 21, 8], 14)
        metadata = data.builder.build(0, common)
        assert (metadata.num_decodes, metadata.num_decode_tokens) == (2, 6)
        query = torch.randn(14, 12, 256, device=data.key.device, dtype=data.key.dtype)
        key = torch.cat([data.key[0, 29:33], data.key[1, 19:21], data.key[2, :8]])
        value = torch.cat(
            [data.value[0, 29:33], data.value[1, 19:21], data.value[2, :8]]
        )
        output = torch.empty_like(query)
        result = data.impl.forward(
            data.layer, query, key, value, data.kv_cache, metadata, output=output
        )
        expected = self._mtp_reference(data, query, common)
        expected[6:] = torch.nn.functional.scaled_dot_product_attention(
            query[6:].transpose(0, 1),
            key[6:].transpose(0, 1),
            value[6:].transpose(0, 1),
            is_causal=True,
            scale=data.impl.scale,
            enable_gqa=True,
        ).transpose(0, 1)
        assert result.data_ptr() == output.data_ptr()
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)

    def test_mtp_forward_graph_replay_updates_device_boundaries(self, mtp_backend):
        """Replay a padded graph with changing lengths and zero-query requests."""
        data = mtp_backend
        common = self._mtp_common_metadata(data, [5, 5, 5], [33, 31, 20], 16)
        metadata = data.builder.build_for_cudagraph_capture(common)
        torch.testing.assert_close(
            metadata.seq_lens, torch.full_like(metadata.seq_lens, 5)
        )
        query = torch.randn(16, 12, 256, device=data.key.device, dtype=data.key.dtype)
        output = torch.empty_like(query)
        workspace = data.manager._current_workspaces[0]
        workspace_ptr = workspace.data_ptr()

        def forward():
            return data.impl.forward(
                data.layer,
                query,
                data.key,
                data.value,
                data.kv_cache,
                metadata,
                output=output,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()

        for query_lengths, seq_lengths in [
            ([2, 4, 0], [17, 33, 0]),
            ([5, 1, 2], [32, 21, 33]),
            ([0, 0, 0], [0, 0, 0]),
        ]:
            updated = self._mtp_common_metadata(data, query_lengths, seq_lengths, 16)
            metadata.query_start_loc.copy_(updated.query_start_loc)
            metadata.seq_lens.copy_(updated.seq_lens)
            query.normal_()
            graph.replay()
            num_real_tokens = sum(query_lengths)
            if num_real_tokens:
                expected = self._mtp_reference(data, query[:num_real_tokens], updated)
                torch.testing.assert_close(
                    output[:num_real_tokens], expected, atol=2e-2, rtol=2e-2
                )
            torch.accelerator.synchronize()
            assert data.manager._current_workspaces[0].data_ptr() == workspace_ptr
