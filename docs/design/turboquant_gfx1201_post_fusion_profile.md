# gfx1201 whole-decode profile after MXFP4 fusion

Date: 2026-09-12. Source revision: `2ed545f872`.
Hardware: Radeon AI PRO R9700, gfx1201, ROCm 7.2.

## Decision

The fused target MXFP4 kernel remains the largest kernel optimization target
after including the MTP drafter and target sampling. It accounts for
66.65--67.16% of all decode kernel time in the two captured workloads. The
drafter accounts for 13.44--13.53%; approximately 70.5% of its device time is
the vocabulary projection.

Activation QDQ and GDN state/conv work each account for approximately 1% of
decode kernel time. The target attention/KV/RoPE kernels account for
0.96--1.57%. These measured shares favor further work on the fused linear
kernel, followed by the shared vocabulary projection if another kernel
target is needed. Any candidate still needs a normal, unprofiled model
comparison because kernel time is only part of request latency.

## Capture and accounting

The served model is `qwen38-27b-tq-mtp`, whose loaded architecture is
`Qwen3_5ForConditionalGeneration`. Both captures use the existing tokenized
128- and 3072-token prompts, greedy sampling, 64 generated tokens, ignored EOS,
unique cache salts, one request at a time, eager V2 execution, MTP with two
draft tokens, and adaptive verification disabled. The gfx1201 K8/V4 attention
and MXFP4 switches are enabled. The established local SDPA-prefill and
unrelated-warmup workarounds are retained.

Each workload is run once before capture. CPU-only `record_function` wrappers
mark target forward, target sampling/logits, the complete drafter proposal,
draft forward, and draft sampling/logits. They call the original methods
without changing tensors or adding device synchronization. The profiler
records shapes, disables stack collection and frontend profiling, and writes
an uncompressed trace for each request.

Every GPU kernel is assigned through its runtime launch correlation ID and
the containing CPU scope on that thread. Nested scopes are used for diagnosis;
the tables below use disjoint ownership. Device timestamp overlap with a CPU
scope is not used to infer ownership. All 215,182 kernels across both traces
have a matching launch. End-of-request calls without a model forward are
accounted for separately and launch no kernels in these captures.

There are 26 and 28 target decode forwards respectively, plus one target
prefill forward per request. The complete drafter proposal, including its
final output-limit tail work, is included in the device totals.

## Decode breakdown

Percentages use the sum of decode kernel durations, not request wall time.

| Exclusive component | 128-token prompt, ms (%) | 3072-token prompt, ms (%) |
| --- | ---: | ---: |
| Target fused MXFP4 | 1464.98 (67.16%) | 1558.93 (66.65%) |
| Target 96-wide emulation linear | 43.87 (2.01%) | 46.92 (2.01%) |
| Target activation QDQ | 21.33 (0.98%) | 22.70 (0.97%) |
| Target GDN state/conv | 21.62 (0.99%) | 23.16 (0.99%) |
| Target attention/KV/RoPE | 20.98 (0.96%) | 36.76 (1.57%) |
| Other target forward kernels | 203.69 (9.34%) | 217.29 (9.29%) |
| Target vocabulary projection and sampling | 105.73 (4.85%) | 112.48 (4.81%) |
| Entire MTP drafter | 293.07 (13.44%) | 316.48 (13.53%) |
| Runner preparation and postprocessing | 6.00 (0.28%) | 4.28 (0.18%) |
| Total decode | 2181.27 (100%) | 2338.98 (100%) |

GDN here means the recurrent update and causal convolution, excluding its
linear projections and normalization. Those projections are already counted
under the linear categories. Attention includes the specialized stage-1 and
stage-2 kernels, cache stores, and RoPE; it excludes QKV/output projections.
Other target forward work includes the native norm arithmetic, casts, copies,
activations, and indexing. The drafter row includes its own attention and
normalization, so these costs are not added again in target rows.

The target forward as a whole is 81.44--81.48% of decode device work. The fused
linear accounts for 81.80--82.47% of target-forward kernel time. There are
6656 and 7168 fused calls, respectively, or 256 per target decode forward.

The small fallback is visible: there are 1248/1344 weight-dequant calls and
the same number of small BF16 GEMMs, corresponding to 48 projections per
forward. Weight dequant alone costs 5.19/5.45 ms. Thus the original fusion
removes large weight materializations while retaining the intended narrow-N
fallback.

Within decode, draft vocabulary projection costs 206.95/222.87 ms across
52/56 calls. Target vocabulary projection costs 103.67/111.63 ms across
26/28 calls. Combined, these projections consume about 14.3% of decode
kernel time. Draft forward and proposal bookkeeping together account for
only the remaining approximately 29.5% of drafter device time.

The ROCm trace reports zero-valued launch grids for the fused Triton kernel.
Consequently, this capture establishes aggregate fused-kernel priority but
does not identify which of its six dense dimensions is the best tuning
target. Use the existing fixed-buffer model-shape benchmark for that choice.

## Prefill and profiler overhead

The whole-request kernel totals, including the drafter, are:

| Prompt tokens | Prefill kernels | Decode kernels | Prefill share |
| ---: | ---: | ---: | ---: |
| 128 | 309.10 ms | 2181.27 ms | 12.41% |
| 3072 | 3338.18 ms | 2338.98 ms | 58.80% |

The 3072-token request is therefore materially affected by prefill even though
the decode optimization target is clear. These normal large-M prefill calls
use emulation because their row count exceeds four. The implementation's
dispatch contract is row count, not semantic prefill/decode phase.

Profiling significantly increases observed latency. After capture, one
warmup and three measured repetitions per context give median normal decode
rates of 24.36/22.03 tok/s in this process; the profiled requests give
13.47/12.60 tok/s. These are profiler-overhead checks in an unlocked-clock
session, not evidence of an additional implementation speedup over the prior
fusion report.

The decode device envelopes span 4785.90/4882.94 ms, while kernel unions cover
2181.26/2338.90 ms. Their 2604.63/2544.04 ms gaps include launch gaps, tracing
overhead, synchronization, and activity outside the captured kernel category.
They must not be attributed entirely to normal Python overhead or treated as
a measured benefit available from graph capture. The profiler's inclusive
GPU annotations also contain intervals rather than additive kernel costs;
the correlation-based kernel sums are the basis for the decision.

## Validation and artifacts

All 12 requests across warmup, profiling, and normal repetitions produce the
same 64-token hash as the preserved fused-backend baseline. For each pair of
contexts, server counters agree: 53 speculative steps, 106 drafted tokens,
73 accepted tokens, with 42/31 acceptances by draft position. The profiled
pair matches the unprofiled pair; the final counters are exactly six times
the pair counters.

The review follow-up also adds `(M, N, K) = (3, 513, 160)` to the existing GPU
correctness test. It exercises partial N and K tiles against independent
dequantization plus `F.linear`, with zero tolerance. The expanded suite passes
all nine cases:

```bash
docker exec \
  -e PYTHONPATH=/workspace/vllm:/tmp/tq-venv/lib/python3.12/site-packages \
  -w /workspace/vllm tq-e2e-current \
  /tmp/tq-mtp-eval/.venv/bin/python -m pytest \
  tests/kernels/quantization/test_mxfp4_gfx1201.py -q
```

Raw traces, profiler tables, request records, Prometheus counters, exact
capture/analysis helpers, and `summary.json` are preserved under
`/tmp/tq-post-fusion-profile.nzulGs`. The summary audit checks all token hashes,
counter ratios, kernel counts, complete correlations, and agreement between
the exclusive cells and phase totals. The profiling server is stopped.

The subsequent
[full-vocabulary projection audit](turboquant_gfx1201_vocab_projection.md)
finds that the shared BF16 head is already bounded by R9700 memory bandwidth,
so it retains the existing `wvSplitK` implementation.
