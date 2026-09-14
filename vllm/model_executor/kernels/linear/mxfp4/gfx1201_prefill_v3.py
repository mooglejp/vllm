# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inactive interfaces for docs/design/gfx1201_prefill_rearchitecture.md.

No plugin registration, environment lookup, tensor allocation, JIT compilation,
or device discovery occurs here. Direct calls to unfinished operations fail.
Do not import this module from production selection until R4 is qualified.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class WeightLayout(str, Enum):
    CHECKPOINT = "checkpoint_n_k2"
    FRAGMENT_V1 = "fragment_v1"


class WeightNumerics(str, Enum):
    GROUP_SCALED = "group_scaled"
    FOLDED_E4M3 = "folded_e4m3"


@dataclass(frozen=True)
class PrefillCall:
    """CPU-known metadata. M alone must never establish phase or ownership."""

    arch: str
    phase: str
    owner: str
    m: int
    n: int
    k: int
    input_dtype: str
    has_bias: bool = False
    last_dim_contiguous: bool = True
    tensor_parallel_size: int = 1
    graph_capture: bool = False


def structural_rejection(call: PrefillCall) -> str | None:
    """Check the planned narrow profile; None is NOT permission to launch."""
    if call.arch != "gfx1201":
        return "architecture"
    if call.phase != "prefill" or call.owner != "target":
        return "phase_or_owner"
    if call.tensor_parallel_size != 1 or call.graph_capture:
        return "parallelism_or_capture"
    if call.m < 64 or call.n < 512 or call.k <= 0 or call.k % 64:
        return "shape"
    if call.input_dtype != "bfloat16" or call.has_bias:
        return "dtype_or_bias"
    if not call.last_dim_contiguous:
        return "stride"
    return None


def can_use_gfx1201_prefill_v3(call: PrefillCall) -> bool:
    """R4 placeholder. Deliberately false, even for a matching profile."""
    _ = call
    return False


@dataclass(frozen=True)
class PreparedWeights:
    """R2 ownership contract; never overwrite the tensors used by decode.

    original_packed: uint8 [N,K/2], low nibble at even K.
    original_scales: uint8 [N,K/32], OCP E8M0.
    packed/scales: derived buffers with explicit layout metadata.
    A second packed tensor consumes real persistent VRAM and must be charged.
    Folding changes numerics and is not implied by layout_version.
    """

    original_packed: torch.Tensor
    original_scales: torch.Tensor
    packed: torch.Tensor
    scales: torch.Tensor
    reference_exponents: torch.Tensor | None
    layout: WeightLayout
    numerics: WeightNumerics
    n: int
    k: int
    layout_version: int
    source_digest: str


@dataclass(frozen=True)
class PrefillWorkspace:
    """R4 buffers owned by one execution lane; addresses survive async work."""

    activation_bytes: torch.Tensor
    activation_scales: torch.Tensor
    output: torch.Tensor
    scratch: torch.Tensor | None
    row_capacity: int
    k_capacity: int
    n_capacity: int


def prepare_weights(
    *,
    packed: torch.Tensor,
    scales: torch.Tensor,
    layout: WeightLayout,
    numerics: WeightNumerics,
) -> PreparedWeights:
    """R2/R3: load-time preparation; lossless packing and folding stay separate.

    Verify the pack/unpack round trip and source ownership before publishing
    the prepared object. No data_ptr-only memoization or in-place repacking.
    """
    raise NotImplementedError("R2/R3 weight preparation is not implemented")


def allocate_workspace(
    *, row_capacity: int, k_capacity: int, n_capacity: int, device: torch.device
) -> PrefillWorkspace:
    """R4: allocate before capture, with explicit per-lane memory accounting."""
    raise NotImplementedError("R4 workspace allocation is not implemented")


def quantize_activation(
    *, x: torch.Tensor, output_bytes: torch.Tensor, output_scales: torch.Tensor
) -> None:
    """R2: BF16 [M,K] to native E4M3 bytes and FP32 [M] scales.

    R0 fixes rounding, saturation, minimum scale and nonfinite-input policy.
    The reference must consume the actual produced FP8 bytes.
    """
    raise NotImplementedError("R2 activation quantization is not implemented")


def launch_raw_fp8(
    *,
    a_bytes: torch.Tensor,
    b_bytes: torch.Tensor,
    output: torch.Tensor,
    variant: str,
) -> None:
    """R1: out-of-tree A[M,K] times B[N,K].T; caller owns all buffers.

    Independently test lane ownership, tail masks, aligned vector accesses,
    K-slab reuse and direct output stores. No MXFP4 decoding or hidden scales.
    """
    raise NotImplementedError("R1 native raw-FP8 mapping is not implemented")


def launch_prefill(
    *,
    x: torch.Tensor,
    weights: PreparedWeights,
    workspace: PrefillWorkspace,
    call: PrefillCall,
) -> torch.Tensor:
    """R2-R4: explicit numerical mode, phase, layout and workspace contract.

    Validate tensor shape/dtype/stride/device, prepared-layout version, source
    ownership and capacity before launch. Never infer layout from shape alone.
    The future plugin delegates ineligible calls to the original kernel.
    """
    raise NotImplementedError("R2-R4 packed prefill linear is not implemented")
