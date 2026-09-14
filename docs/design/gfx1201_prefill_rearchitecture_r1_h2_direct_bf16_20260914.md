# H2 direct-BF16 epilogue follow-up — 2026-09-14

## Scope

This is the single bounded R1 follow-up from commit
`4429f0e468791ea9a958205d81d4509c16d88128`. It keeps H2's input layout,
K=64 staging, WMMA computation order, accumulator type, and grid mapping
unchanged. Only the output epilogue is split into two benchmark-only paths:

- existing H2: FP32 global scratch followed by the existing BF16 postcast;
- H2 direct BF16: the same accumulator is staged through the existing
  wave-sized LDS tile and converted directly to caller-owned BF16 output,
  without the global FP32 scratch.

H1, old A3, production registration, dispatch, thresholds, decode/MTP/K8V4
lanes, R2 MXFP4 fusion, and model integration were not changed.

## R0 coverage clarification

The R0 manifest remains the frozen synthetic-GEMM comparison contract. Its
`r0_coverage_status` now records:

- synthetic GEMM manifest: frozen and measured;
- real-model M/N/K/stride/phase/role/call collection: not collected;
- matched fork/Radiance control execution: not run.

The R1 synthetic benchmark remains valid; these uncollected items must not be
reported as completed model qualification.

## Correctness

The follow-up JSONL contains 17 correctness cases. All variants pass the
existing FP64-byte-oracle BF16 checks, and H2 direct BF16 is bitwise equal to
the existing H2 FP32-scratch-plus-postcast output in all 17 cases. The cases
include basis vectors, 129x129x64 row/column-distinct input, 65x129x65
M/N/K tails, full H1/H2 tiles, and representative rows/columns of all frozen
shape families. N=96 remains a fallback/correctness case rather than a
large-N timing-gate member.

Artifact: [`r1_h2_direct_bf16_measurements`](/tmp/vllm-tq-prefill-rearchitecture-r1-final/docs/design/artifacts/gfx1201_prefill_rearchitecture_r1_h2_direct_bf16_20260914.jsonl).

## Timing

The formal pass used the same GPU and ROCm 7.14 environment, fixed inputs and
buffers, 5 warmups, 20 order-rotated samples, and a 64 MiB flush. H2 direct
BF16 includes its direct epilogue in the timed operation. The primary baseline
is the same-session pre-expanded BF16 `torch.mm`.

### M=256, N>=512 (median microseconds; ratio is BF16 / candidate)

| shape | N | K | calls | BF16 | H2 postcast | H2 direct BF16 | H2/BF16 | direct/BF16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5120 | 6144 | 64 | 327.060 | 324.681 | 309.860 | 1.0073x | 1.0555x |
| 1 | 34816 | 5120 | 64 | 982.502 | 1398.183 | 1399.483 | 0.7027x | 0.7020x |
| 2 | 5120 | 17408 | 64 | 646.281 | 829.782 | 813.622 | 0.7789x | 0.7943x |
| 3 | 16384 | 5120 | 48 | 631.461 | 654.961 | 726.902 | 0.9641x | 0.8687x |
| 5 | 14336 | 5120 | 16 | 569.841 | 635.061 | 722.782 | 0.8973x | 0.7884x |

Calls-weighted totals over the five required shape IDs `[0, 1, 2, 3, 5]`:

| method | weighted latency (us) | speedup vs BF16 | gate |
| --- | ---: | ---: | --- |
| BF16 baseline | 164601.567 | 1.0000x | reference |
| H2 postcast | 204968.472 | 0.8031x | fail |
| H2 direct BF16 | 207925.596 | 0.7916x | fail |

The direct epilogue is approximately 1.44% slower than the H2 postcast path
in this weighted session. It improves shape 0 and shape 2, but is slower on
shape 3 and shape 5; the shape-weighted result does not improve. Neither H2
variant reaches the unchanged 1.0x BF16 feasibility gate.

## Shape-set defense

The weighted gate now requires the observed `shape_index` set to equal
`[0, 1, 2, 3, 5]` with no duplicates. The formal artifact records
`shape_id_set_complete: true`; a partial, duplicated, or substituted set is
incomplete and cannot pass based on count alone.

## Generated-object observation

The direct-BF16 build and extracted gfx1201 code object are recorded in
[`r1_h2_direct_bf16_object`](/tmp/vllm-tq-prefill-rearchitecture-r1-final/docs/design/artifacts/gfx1201_prefill_rearchitecture_r1_h2_direct_bf16_object_20260914.txt).
The H2 full direct-BF16 kernel uses 44 SGPR, 101 VGPR, 33,280 LDS bytes,
no spills, and wave32. The existing H2 full FP32-output kernel in the same
build uses 40 SGPR, 101 VGPR, 25,088 LDS bytes, no spills, and wave32. Static
disassembly observations show the direct full kernel retains 32 FP8 WMMA
operations and the same global FP8 load counts, while adding the sequential
LDS epilogue and BF16 global stores. These are observations, not a causal
performance attribution.

## Decision

The one allowed H2 epilogue correction is complete and fails the unchanged
same-session BF16 gate. The prior R1 H1/H2 result remains preserved; H2 is a
useful diagnostic mapping but is not an accepted production candidate. Stop
further H2 micro-tuning here. Do not start R2, MXFP4 fusion, production
integration, or another tile/mapping search.
