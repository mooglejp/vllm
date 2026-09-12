# gfx1201 TurboQuant full-decode graph validation

Date: 2026-09-12. Source: `000ed0f3f1`. Hardware: Radeon AI PRO R9700,
gfx1201, ROCm 7.2. Production defaults are unchanged by this evaluation.

## Result

Whole-model `FULL_DECODE_ONLY` graph capture and replay passes for the opt-in
gfx1201 TurboQuant MTP route in the tested single-request, concurrent-request,
and 128-32K context workloads. This is stronger than the earlier
attention-backend graph test: the target model, two-step MTP drafter, and
decode-side model-runner forward work execute through full graphs.

Across 36 paired graph/eager requests, every output token ID, full token-stream
hash, completion count, prompt hash, and per-request speculative-decoding
summary is exact. Runtime graph metrics report `FULL` for decode token shapes
3, 6, 9, and 12. Prefill and continuation-prefill shapes report `NONE`, as
required by `FULL_DECODE_ONLY`.

This moves the full-decode graph, multi-request, and 32K-context gates to pass
for the tested configuration. The subsequent
[rollout validation](turboquant_gfx1201_rollout_validation.md) adds working MTP
prefix reuse, a 308-case quality comparison, and a 24-minute serving soak. It
also retains opt-in rollout because the default O2 compile/graph policy is not
ready for this route.

## Configuration and method

The server uses the local
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` checkpoint, TP=1, the V2 model runner,
`turboquant_k8v4`, the opt-in gfx1201 K8/V4 path, forced SDPA prefill, two MTP
draft tokens, and adaptive verification disabled. The main decoder uses the
production gfx1201 software-emulated MXFP4 backend. Graph runs use compilation
mode `NONE` and `FULL_DECODE_ONLY`; their controls use `--enforce-eager`.

Requests use greedy sampling, ignore EOS, emit exactly 64 tokens, and carry a
unique cache salt. The one-request and concurrent tests use a 4096-token model
and scheduler limit, four maximum sequences, and 0.90 GPU memory utilization.
The wide-context tests use a 33,024-token model limit, one maximum sequence,
and 0.75 GPU memory utilization.

Each one-request point has one warmup plus three measured requests. The
concurrent workload has four synchronized requests with distinct 128, 256,
512, and 1024-token prompts, one warmup round, and three measured rounds. Wide
points have one warmup plus one measured request. Decode throughput excludes
the first streamed token and prefill. End-to-end throughput includes both.
Clocks were not fixed, so rates are controlled observations rather than formal
performance claims.

## Exactness and throughput

The single-request run captured token shape 3 and paired 12 graph requests with
12 fresh eager requests.

| Prompt tokens | Graph decode tok/s | Eager decode tok/s | Decode speedup | Graph E2E tok/s | Eager E2E tok/s | E2E speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 28.835 | 17.479 | 1.650x | 25.679 | 16.176 | 1.588x |
| 1024 | 29.395 | 18.014 | 1.632x | 20.565 | 14.034 | 1.465x |
| 3072 | 25.799 | 16.207 | 1.592x | 10.797 | 8.594 | 1.256x |

The synchronized four-request run captured shapes 3, 6, 9, and 12. All 16
paired requests are exact. Graph and eager each record 412 speculative steps,
824 draft tokens, and 608 accepted draft tokens. Median aggregate completion
throughput is 30.447 graph versus 25.241 eager tok/s, a 1.206x speedup.

The wide run extends the exact comparison through 32K. The 4K-16K points use a
1024-token scheduler budget. The 32K point uses 512 to bound continuation-
prefill attention workspace.

| Prompt tokens | Prefill chunk | Graph decode tok/s | Eager decode tok/s | Decode speedup | Graph E2E tok/s | Eager E2E tok/s | E2E speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1024 | 27.955 | 17.104 | 1.634x | 9.032 | 7.204 | 1.254x |
| 8192 | 1024 | 29.021 | 18.256 | 1.590x | 4.307 | 3.821 | 1.127x |
| 16384 | 1024 | 23.780 | 20.120 | 1.182x | 1.641 | 1.624 | 1.011x |
| 32768 | 512 | 27.942 | 24.265 | 1.152x | 0.550 | 0.548 | 1.002x |

The diminishing long-context end-to-end gain is expected: continuation
prefill dominates wall time and intentionally remains eager in this graph
mode. The result supports decode graph correctness and benefit; it is not
evidence that graph replay accelerates long prefill.

## Default O2 policy control

A separate control leaves the repository's default optimization policy
untouched. It resolves to `CompilationMode.VLLM_COMPILE` with
`FULL_AND_PIECEWISE` and capture sizes 1-4. The first startup compiles the
backbone in 79.73 seconds and the MTP head in 19.45 seconds, then captures both
piecewise and full graphs. A matched control uses the same compilation mode
with graph mode `NONE`.

All six default-policy graph requests exactly match the compiled/no-graph
control, including output tokens and speculative counters. Runtime metrics
report token shape 3 as `FULL`, so the default graph dispatcher is correct in
these cases.

| Prompt tokens | Default graph decode tok/s | Compiled no-graph tok/s | Speedup | Default graph E2E tok/s | Compiled no-graph E2E tok/s | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8.935 | 7.877 | 1.134x | 8.417 | 7.425 | 1.134x |
| 1024 | 9.190 | 8.122 | 1.131x | 8.292 | 7.401 | 1.120x |
| 3072 | 10.096 | 8.954 | 1.128x | 6.763 | 6.223 | 1.087x |

The compiled and compilation-disabled paths are not numerically interchangeable:
all three deterministic prompts produce different token streams and
speculative-acceptance summaries. More importantly for rollout, default O2
full-graph decode is 61-69% slower than the compilation-disabled
`FULL_DECODE_ONLY` measurements above. The experiment does not attribute the
gap to a specific compiler pass, but it rules out enabling this optimized route
under the current default policy without a separate compiler-path diagnosis.

## Graph memory and serving shapes

| Capture set | Actual graph pool | Estimated graph pool | Available KV cache | KV capacity |
| --- | ---: | ---: | ---: | ---: |
| `[3]`, 4096-token limit | 0.19 GiB | 0.22 GiB | 6.96 GiB | 39,497 tokens |
| `[3, 6, 9, 12]`, 4096-token limit | 0.80 GiB | 1.30 GiB | 5.88 GiB | 33,353 tokens |
| Default O2 `[1, 2, 3, 4]`, 4096-token limit | 1.27 GiB | 1.31 GiB | 5.79 GiB | 32,768 tokens |
| `[3]`, 33,024-token limit | 0.15 GiB | 0.17 GiB | 2.34 GiB | 53,074 tokens |

The four-request capture set costs about 0.61 GiB more graph-pool memory than
the one-request set and correspondingly reduces KV capacity. Capture sizes
should therefore match the intended concurrency instead of being expanded
without a serving need.

The concurrent workload begins all four requests together and uses distinct
prompt lengths. The primary run repeatedly exercises shape 12 and naturally
reaches shape 9; focused two- and three-request follow-ups exercise shapes 6
and 9, while request completion exercises shape 3. All four appear as `FULL`
in runtime metrics. Prefill and mixed prefill work remains eager.

## Long-context memory boundary

The failed configurations isolate a continuation-prefill workspace constraint,
not a graph-replay failure:

- With a 4096-token scheduler budget and 0.90 memory utilization, an 8192-token
  request fails in `TurboQuantAttentionImpl._continuation_prefill` while PyTorch
  SDPA requests 2.93 GiB. The scheduler step is prefill and reports runtime
  `NONE`.
- With a 1024-token budget and 0.75 utilization, 4K, 8K, and 16K pass. The 32K
  request reaches 28,272 computed tokens before SDPA requests 2.68 GiB with
  2.46 GiB free. This is also a `NONE` prefill step.
- Lowering utilization to 0.70 is not viable for this 33,024-token server: only
  0.70 GiB is available for KV while one maximum-length request needs 1.44 GiB.
- Restoring utilization to 0.75 and reducing the scheduler budget to 512 lets
  both 32K graph and eager requests complete twice with exact outputs.

Long-context deployments on this 32 GiB card need to balance KV capacity and
the quadratic SDPA continuation-prefill workspace. Smaller prefill chunks are
the validated configuration control. The failure does not motivate a decode
kernel or graph change.

## Default decision

Keep `VLLM_TQ_GFX1201_K8V4` and
`VLLM_ROCM_USE_GFX1201_MXFP4_GEMM` opt-in for now.

- Draft KV ownership and cross-request prefix reuse now pass. A repeated
  3000-token request reuses 2976 tokens across the full-attention and Mamba
  groups. See the rollout validation for the implementation and live result.
- The broader 308-case GSM8K, HumanEval, and MMLU comparison and the 24-minute,
  484-request serving soak pass without an aggregate quality regression,
  request error, abort, or preemption.
- The repository's default O2 `FULL_AND_PIECEWISE` graph is exact against a
  matched compiled/no-graph control, but changes all three sampled outputs and
  is substantially slower than the compilation-disabled path. The existing
  quality gate does not cover this compiled configuration.
- Native-MXFP4 hardware evaluation is outside this rollout decision. The
  custom backend remains scoped to gfx1201 software emulation.

The prefix-cache, quality, and serving gates are complete. Keep the route
opt-in under the explicitly validated compilation-disabled full-decode policy;
default enablement should follow a compiler-path diagnosis. Full-decode graph
support itself is no longer a blocking gate for this exact profile.

## Artifacts

Raw request JSONL, server logs, runtime graph statistics, failed-run stacks,
and machine-readable summaries are preserved under
`/tmp/tq-full-graph.fbuMq6`. Local launch and comparison helpers are under
`.tools/tq_accuracy_eval/` and `.tools/tq_full_graph/`.
