# gfx1201 full-vocabulary projection audit

Date: 2026-09-12. Starting revision: `551ec16544`.
Hardware: Radeon AI PRO R9700, gfx1201, ROCm 7.2.

## Decision

Keep the existing ROCm `wvSplitK` vocabulary projection. No production kernel
or dispatch change is adopted in this iteration.

The target and MTP drafter share a 2.543 GB BF16 head. The current projection
reads that head at an estimated 635--637 GB/s while AMD specifies 640 GB/s peak
memory bandwidth for the R9700. Changing the `wvSplitK` dispatch-count argument
does not produce a stable improvement, and a direct Triton matrix kernel is
slower and not bitwise equivalent. Preserving the full-vocabulary, BF16-weight
contract leaves no material launch-only speedup to adopt.

This result lowers the priority of more tuning in the same vocabulary-kernel
scope. It does not claim that a different weight representation, reduced-vocab
algorithm, fused sampler, or different hardware could not improve the broader
logits path; those changes are outside this audit.

## Observed production contract

A local observation wrapper records the arguments immediately around
`compute_logits` without replacing the projection. Two single-request runs use
128- and 3072-token prompts and generate 16 tokens each. Both use eager V2,
TP=1, MTP with two draft tokens, the production gfx1201 MXFP4 backend, and the
existing TurboQuant attention configuration.

| Property | Target | Drafter |
| --- | ---: | ---: |
| Decode input | `[3, 5120]` | `[1, 5120]` |
| Decode calls in the two requests | 13 | 26 |
| Prefill input | `[1, 5120]` | `[1, 5120]` |
| Prefill calls | 2 | 4 |
| Weight | `[248320, 5120]` BF16 | shared with target |
| Output | full `[M, 248320]` BF16 | full `[M, 248320]` BF16 |

The inputs and weight are contiguous with unit K stride. The head has no bias,
uses `UnquantizedEmbeddingMethod`, and has `head_dtype=torch.bfloat16`. Its
original and padded vocabulary sizes are both 248320, so this model has no
post-projection padding columns to remove. Target and drafter report the same
weight data pointer in this run.

`LogitsProcessor._apply_head` calls the head's unquantized method. On ROCm this
reaches `rocm_unquantized_gemm`; with skinny GEMM enabled, a contiguous BF16
weight, no bias, and one to five rows, it dispatches `wvSplitK`. Prior correlated
profiles identify the same `wvSplitK_hf_sml_` kernels inside both logits scopes.

The runtime reports 32 multiprocessors for gfx1201, so vLLM passes 32 as the
default `CuCount` argument. AMD documents 64 physical compute units; the two
numbers use different reporting conventions and must not be conflated.

## Direct Triton candidate

The first experiment uses the live shared head and the first observed M=1 and
M=3 inputs. Twelve Triton configurations cover N tiles 32--256, K tiles
32--128, and four or eight warps. Each program computes a 16-row padded output
tile with FP32 accumulation and writes the full BF16 logits tensor. Candidate
output buffers are allocated before timing. Measurements use three warmups,
15 samples, a 64 MiB L2 flush, and rotating operation order.

| Rows | Existing `wvSplitK` | Best Triton | Triton / Existing latency |
| ---: | ---: | ---: | ---: |
| 1 | 4069.8 us | 4116.5 us | 1.011x |
| 3 | 4094.1 us | 4139.9 us | 1.011x |

The Triton sweep's server used the emulated MXFP4 linear backend. That changes
the surrounding model path, not the vocabulary projection or its BF16 head.
The separately measured `wvSplitK` experiment below uses the production MXFP4
backend.

All Triton configurations produce the same output as each other, but not the
same output as `wvSplitK`. On the captured input, maximum absolute difference
is 0.03125; RMSE is 0.000113 at M=1 and 0.000211 at M=3. This is consistent
with a different reduction order, but the artifact only establishes the
observed difference. The candidate is rejected on both speed and exactness.

## `wvSplitK` dispatch-count sweep

The second experiment retains the exact production kernel and changes only its
`CuCount` argument. It runs on the live model weight and input with the
production MXFP4 backend enabled. Counts 32, 40, 48, 56, 60, 64, 72, 80, and
96 use five warmups, 25 cold-L2 samples, and rotating operation order.

Every count is bitwise exact against the default for the complete output. The
small timing differences do not select a reproducible replacement:

| Input | Default 32 | Best live count | Best live | Gain |
| --- | ---: | ---: | ---: | ---: |
| M=1 | 3994.5 us | 48 | 3992.5 us | 1.0005x |
| M=3 | 4004.8 us | 80 | 4003.3 us | 1.0004x |

Two standalone runs use a seeded random BF16 weight of the same shape and
preserve every raw timing sample. Their best M=1 count remains 48, but the best
M=3 count changes from the live run's 80 to 60. In the final rerun, gains over
the default are only 0.042% and 0.057%, and the p10--p90 intervals overlap.

| Input | Default 32 | Best standalone | Estimated bandwidth |
| --- | ---: | ---: | ---: |
| M=1 | 3994.4 us | 3992.7 us, count 48 | 637.0 GB/s |
| M=3 | 4006.1 us | 4003.8 us, count 60 | 635.5 GB/s |

The traffic estimate counts one read of the contiguous BF16 weight and input
plus one BF16 output write. It is 2,543,303,680 bytes at M=1 and 2,544,317,440
bytes at M=3. It does not claim a hardware-counter measurement. The comparison
to the R9700's [official 640 GB/s peak specification](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)
is a roofline sanity check, not a guarantee that every byte bypasses cache.

At the observed traffic rate, even reaching the advertised peak would improve
one projection by only about 0.5%. Applied to the earlier 14.3% combined
target/drafter vocabulary share, the idealized decode-kernel gain is below
0.1%. The unstable microsecond differences therefore do not justify a new
model-specific dispatch branch.

## Public benchmark

`benchmarks/kernels/benchmark_gfx1201_vocab_projection.py` reproduces the
observed M=1/M=3, N=248320, K=5120 contract with a seeded random BF16 weight.
It records raw samples, median, p10/p90, estimated traffic, and full-output
correctness for each `wvSplitK` count. Input generation, allocation of the
weight, warmup, and correctness checks are outside the timed interval. The
timed callable is the production `wvSplitK` wrapper, whose internal output
allocation cannot be separated from its kernel launch. GPU events measure the
device interval and do not claim an isolated host-allocation cost.

```bash
docker exec \
  -e PYTHONPATH=/workspace/vllm:/tmp/tq-venv/lib/python3.12/site-packages \
  -w /workspace/vllm tq-e2e-current \
  /tmp/tq-mtp-eval/.venv/bin/python \
  benchmarks/kernels/benchmark_gfx1201_vocab_projection.py \
  --output /tmp/tq-vocab-public-benchmark-final.json
```

All tested `wvSplitK` counts are bitwise exact for both row counts. Direct
`torch.nn.functional.linear` is slower and differs from `wvSplitK`, confirming
that substituting a generic GEMM would also require a separate numerical
decision rather than token-only validation.

## Validation boundary and artifacts

No production code path changes, so a candidate same-load model comparison,
MTP acceptance comparison, and 768-vector replay are intentionally not claimed.
Those remain adoption gates for any future candidate. The accepted MXFP4
kernel and all existing generated outputs are unchanged.

The observation contracts, captured decode inputs and logits, live-weight
Triton and `CuCount` sweeps, generated ISA, standalone raw samples, and exact
local helpers are preserved under `/tmp/tq-vocab-projection.7FvtiR`. The failed
first observation attempt is excluded: its local wrapper resolved the outer
conditional-generation module instead of the delegated language model and
stopped during startup before serving a request.
