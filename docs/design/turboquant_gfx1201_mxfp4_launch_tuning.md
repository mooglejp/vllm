# gfx1201 MXFP4 launch-configuration tuning

Date: 2026-09-12. Starting point: 8334b6b8d1. Hardware: gfx1201 / ROCm 7.2.

## Result

The opt-in software-fused MXFP4 decode kernel now uses a 16x64 output tile,
K block 128, four wave32 warps, and one pipeline stage. The prior launch used a
16x32 tile, two warps, and one stage. No decoding expression, accumulation
type, activation QDQ, weight layout, or dispatch boundary changes in this
iteration.

Over the five eligible Qwen3.5-27B dense shapes, production-call-count-weighted
cold-L2 kernel time improves by 1.074x at M=1 and 1.072x at M=3. A
profiler-free comparison in one loaded MTP server improves median decode rate
by 1.007--1.008x across 128, 1024, and 3072 prompt tokens. The smaller model
gain is reported as measured; it is not extrapolated from the standalone
kernel result.

All random-shape checks, a captured real model input, generated token hashes,
MTP counters, and 768 target-logit vectors remain exact.

## Benchmark correction

The original benchmark called the linear wrapper inside each timed interval.
That wrapper creates its output with torch.empty, so it was not a fixed-output
kernel-only measurement.

The updated benchmark reports two distinct operations:

- direct configurations launch the kernel into a tensor allocated before
  warmup and timing;
- fused_wrapper retains wrapper-level timing, including output allocation.

The public shape sweep feeds every direct candidate the same generated BF16
activation, packed weight, and scale tensors. It does not run activation QDQ.
A separate captured-input check uses the QDQ-complete activation observed in
the model. Compilation, allocation, and correctness checks are outside the
direct timing interval. Each timed operation follows a 64 MiB device-side L2
flush. Operation order rotates every sample so one candidate does not always
occupy the same order position.

## Production shape weights

An opt-in CPU-side diagnostic counted calls while running three 64-output-token
MTP requests. Decode target calls all had M=3. Dividing the counts by the 80
observed target forwards gives the stable per-forward shape weights:

| N x K | Calls per target forward |
| ---: | ---: |
| 5120 x 6144 | 64 |
| 34816 x 5120 | 64 |
| 5120 x 17408 | 64 |
| 16384 x 5120 | 48 |
| 14336 x 5120 | 16 |
| **Eligible total** | **256** |
| 96 x 5120 | 48, emulation fallback |

The same architecture-derived weights are used for M=1 standalone decode
estimates. M=2 and M=4 are correctness and adoption checks; they were not
observed in this MTP request trace.

## Pipeline-stage sweep

The first sweep changes only num_stages, retaining BLOCK_M=16, BLOCK_N=32,
BLOCK_K=128, and two warps. Every one of the five shapes is exact at M=1 and
M=3.

| Stages | Weighted M=1 | vs stage 1 | Weighted M=3 | vs stage 1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64.92 ms | 1.000x | 65.26 ms | 1.000x |
| 2 | 77.52 ms | 0.838x | 78.06 ms | 0.836x |
| 3 | 77.33 ms | 0.840x | 80.36 ms | 0.812x |

Stage 1 uses 132 VGPRs, 105 SGPRs, 4 KiB shared staging, and no private
segment. Stage 2 increases to 191 VGPRs and 12 KiB shared staging. Stage 3
reaches 256 VGPRs and 16 KiB shared staging and spills to a 96-byte private
segment. The generated ISA and measured time both reject deeper pipelining.

## N tile and warp sweep

With stage 1 selected, the second sweep covers BLOCK_N={32,64,128} and
num_warps={2,4} while keeping BLOCK_M=16 and BLOCK_K=128 fixed. All 60
shape/row/configuration combinations are bitwise exact. The weighted result is:

| BLOCK_N | Warps | M=1 weighted | M=3 weighted |
| ---: | ---: | ---: | ---: |
| 32 | 2 | 65.07 ms | 65.64 ms |
| 32 | 4 | 189.27 ms | 190.26 ms |
| 64 | 2 | 84.55 ms | 85.07 ms |
| **64** | **4** | **60.62 ms** | **61.26 ms** |
| 128 | 2 | 295.86 ms | 291.54 ms |
| 128 | 4 | 99.33 ms | 100.48 ms |

For the M=3 production row count, the selected configuration changes median
direct time as follows:

| N x K | Calls | 32x2 | 64x4 | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 5120 x 6144 | 64 | 139.12 us | 129.68 us | 1.073x |
| 34816 x 5120 | 64 | 400.08 us | 369.84 us | 1.082x |
| 5120 x 17408 | 64 | 276.54 us | 273.16 us | 1.012x |
| 16384 x 5120 | 48 | 211.44 us | 187.02 us | 1.131x |
| 14336 x 5120 | 16 | 205.04 us | 176.74 us | 1.160x |

The selected ISA uses 133 VGPRs, 105 SGPRs, 4 KiB shared staging, and no
private segment or scratch. This is almost the same register footprint as the
prior 132-VGPR launch.

## Correctness

The adoption checks cover:

- all five eligible model dimensions at M=1, 2, 3, and 4;
- independent packed-weight dequantization followed by F.linear;
- masked K and N tails, including N=513;
- E8M0 raw scale values 0, 1, 127, 254, and 255;
- one captured QDQ-complete model input with shapes [3,5120],
  [16384,2560], and [16384,160].

For the captured input, the old launch, selected launch, and independent
dequantize-plus-linear reference produce the same BF16 tensor bit for bit.

A canonical-proposal MTP replay uses 12 prompts and 64 output positions per
prompt. The selected launch matches all 768 prior full-FP32 target-logit
SHA-256 values, and all 323 target input batches are equal. Generated token
hashes are identical in every timing request.

## Same-session model measurement

The old and selected launches are switched inside one loaded eager MTP server.
Both paths traverse the same diagnostic Python wrapper; its one-time real-input
capture occurs in an excluded warmup. Medians combine five measured repetitions
per context for the old launch and seven for the selected launch:

| Prompt tokens | Decode old | Decode selected | Speedup | E2E old | E2E selected |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 17.354 tok/s | 17.494 tok/s | 1.008x | 16.067 | 16.183 |
| 1024 | 17.895 tok/s | 18.046 tok/s | 1.008x | 13.934 | 14.049 |
| 3072 | 16.087 tok/s | 16.204 tok/s | 1.007x | 8.555 | 8.603 |

One bounded nine-request old interval records 234 speculative steps, 468 draft
tokens, and 336 accepted draft tokens. Two identical nine-request selected
intervals record exactly twice each count: 468, 936, and 672. This verifies that
the launch change does not alter MTP acceptance.

GPU clocks were not locked. The small model-level difference is consistent
across prompt lengths and repeated blocks, but remains specific to
single-request eager decoding on this R9700. It is not a continuous-batching,
graph-mode, or native-MXFP4 result.

## Scope

The selected geometry stays behind
VLLM_ROCM_USE_GFX1201_MXFP4_GEMM, which remains disabled by default. This
iteration intentionally excludes Split-K, a different MXFP4 decoding formula,
weight reordering, activation-QDQ fusion, prefill changes, and the 96-wide
fallback. The next independent decode target remains the shared target/drafter
vocabulary projection.

That follow-up is recorded in the
[full-vocabulary projection audit](turboquant_gfx1201_vocab_projection.md).
