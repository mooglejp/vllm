// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// INACTIVE: docs/design/gfx1201_prefill_rearchitecture.md, R1-R3.
// Not in CMake, _rocm_C, op registration, or a production import path.
// This file intentionally has no kernel, no no-op launcher, and no bindings.
//
// R1-MAPPING: Derive lane-to-A/B/C ownership from supported AMD interfaces.
// Verify basis/row/column-distinct tests and M/N/K tails independently.
// Eight-wave designs must not assume unsupported rocWMMA cooperative APIs.
//
// R1-STAGING: H1/H2 use K=64 slabs, aligned wide transfers where valid,
// explicitly padded LDS, and multiple matrix instructions per staged slab.
// Keep required synchronization. Do not copy rejected K=16 scalar staging.
//
// R1-EPILOGUE: Register accumulators map directly to output coordinates.
// No full output-tile shared-memory array. Debug FP32 output is separate from
// timed BF16 production-contract output. No whole-model weight expansion.
//
// R2-PACKED: Add independently specified, reversible packed layout only after
// R1 qualification. Verify source ownership and preserve canonical weights.
// Group-scaled weights use group-local partials plus a separate total; never
// rescale accumulated contributions from earlier groups.
//
// R3-FOLDING: Optional, separate numerical mode. Generate transformations from
// the format specification and exhaustively test bytes/exponents/subnormals.
// Do not copy Radiance tables or represent folding as lossless byte packing.
//
// R4-BINDING: A later reviewed commit adds fake/meta support, explicit output
// and scratch ownership, current-stream handling and device/layout checks.
// Do not enable production from the benchmark build.
