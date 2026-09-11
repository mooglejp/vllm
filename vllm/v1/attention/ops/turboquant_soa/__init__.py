# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# HIP-free SoA Triton subset for the FlyDSL TurboQuant decode path.
#
# gfx1201 K8/V4 D=256 GQA=6 implementation scaffold:
#   triton_turboquant_decode_gfx1201_k8v4.py
# Design/phase gates:
#   docs/design/turboquant_gfx1201_k8v4_mtp.md
#
# The scaffold is intentionally not imported here. Wire it only after the
# Phase 2 single-token correctness gates pass; this keeps runtime behavior
# unchanged while the implementation is incomplete.
