# gfx1201 TurboQuant MTP small accuracy gate

Date: 2026-09-12. Source: `7700b69453`. The model, kernel, and serving
defaults are unchanged by this evaluation.

## Result

The small model-quality gate passes for the opt-in gfx1201 TurboQuant MTP
implementation. Native non-MTP (A) and actual two-token MTP (C) have the same
aggregate scores on 104 seeded reasoning, code, and general-knowledge cases:

| Task | Cases | A | C | Semantically equal |
| --- | ---: | ---: | ---: | ---: |
| GSM8K exact answer | 32 | 18 | 18 | 23 |
| HumanEval-derived functional tests | 32 | 32 | 32 | 32 |
| MMLU exact choice | 40 | 31 | 31 | 40 |

This is evidence against a large quality regression, not a statistically
powered model evaluation. It qualifies the existing implementation for
continued **opt-in** use and performance tuning. Default enablement remains
deferred pending broader task coverage and production-serving validation.

Two integration checks strengthen the result:

- Across 12 prompts and 768 validated output positions, an MTP-shaped
  target-only replay without constructing or executing the draft model (B)
  has the same SHA-256 for every full FP32 target-logit vector as actual MTP
  (C). All 323 target input batches also match.
- Four real layer-63 Q/KV snapshots, spanning sequence lengths 111 through
  1777, reproduce their live outputs bitwise. Per-row serial attention and
  packed query widths 1, 2, and 4 are bitwise equal on all 12 query rows.

As in the preceding diagnosis, B and C share the production packed target
kernel. B/C equality is therefore an integration control, not an independent
kernel reference. The real-QKV replay includes both the legacy one-row kernel
and the separate generic TurboQuant reader.

## Configuration

The evaluation uses `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` on gfx1201 / ROCm 7.2,
TP=1, the eager V2 runner, Quark MXFP4 emulation for the main decoder, BF16
MTP weights, `turboquant_k8v4`, forced SDPA prefill, and the opt-in specialized
decode kernel. Generic warmup is disabled equally in A and C because the
unrelated CK FlashAttention warmup crashes in this environment. No graph-mode,
multi-request load, or prefix-cache performance conclusion is made.

A is ordinary native non-MTP scheduling. C uses two MTP draft tokens with
adaptive verification disabled. Sampling is greedy. The suite is generated
with seed 1201 and has SHA-256
`0100adcf971acd3721fee109476ecf64ca59e378a2a57204413ec4451acfc12c`.
Prompt lengths range from 79 to 1745 tokens and total 40,252 tokens.

The suite contains:

- 32 concise, non-thinking GSM8K prompts with at most 128 output tokens;
- 32 non-thinking HumanEval-derived prompts with at most 384 output tokens;
- eight five-shot questions from each of `conceptual_physics`,
  `high_school_world_history`, `professional_psychology`, `business_ethics`,
  and `college_computer_science`, with at most 16 output tokens.

Dataset byte hashes and the exact selected prompts are retained in the local
manifest and suite artifacts. HumanEval candidates are executed separately in
disposable containers with networking disabled, a read-only root filesystem,
no Linux capabilities, `no-new-privileges`, resource limits, and no workspace
mount. This is intentionally a small gate rather than a substitute for the
project's full evaluation infrastructure.

## Paired task results

GSM8K token streams are identical for 20/32 cases and extracted answers for
23/32. Each mode uniquely answers two cases correctly, leaving the same 18/32
aggregate score. MMLU outputs and extracted answers match for all 40 cases;
both score 31/40. HumanEval generated source is identical for 14/32 cases, but
both modes pass all 32 selected functional tests. C reaches the 384-token limit
on `HumanEval/108`; that candidate still passes its tests.

The C run records 1,568 speculative steps and 3,136 proposed draft tokens.
It accepts 2,955 draft tokens: 1,514/1,568 at position zero and 1,441/1,568 at
position one. Including the target bonus token, the mean emitted length per
verification step is 2.8846 tokens. These counters describe this mixed task
suite and should not be generalized to unrelated prompts.

The A and C suite runs emitted 4,998 and 4,499 tokens respectively. Their raw
wall times are not a controlled throughput comparison because output counts,
stop positions, and per-request lengths differ. The earlier fixed-output,
warmed single-request measurements remain the performance baseline.

## Multiprompt B/C target control

The target-only comparison selects four prompts per task with context lengths
79, 83, 92, 115, 154, 168, 212, 302, 374, 561, 825, and 1745. C first runs
naturally for 64 output tokens per prompt. B then replays C's exact proposals
and output trace while preserving the target's speculative metadata, packed
query shape, block size, physical cache geometry, and prefill schedule.

B does not construct or execute the MTP model. Before forced sampling, every
valid target row is converted to FP32 and hashed over the entire vocabulary.
All 768 paired hashes match. The comparison also checks the serialized target
input batches; all 323 match. There are no ignored mismatches. Rows beyond an
incorrect speculative prefix are excluded before pairing.

This extends the earlier single-prompt `B == C` result over three task families
and a 22x context-length range. It excludes a required numerical side effect
from loading or executing the drafter in these cases, but it does not validate
the target kernel independently because that path is intentionally shared.

## Real-QKV attention replay

Layer-63 snapshots are captured from actual C verification batches at four
contexts. Each contains three BF16 query rows, 24 query heads, four KV heads,
head size 256, the exact compressed K8/V4 SoA cache bytes, physical block
mapping, and the live output. The replay uses 32 KV splits.

| Context | Sequence length | Query rows | Live replay | Serial vs packed 1/2/4 |
| ---: | ---: | ---: | ---: | ---: |
| 79 | 111 | 3 | Bitwise | Bitwise |
| 374 | 406 | 3 | Bitwise | Bitwise |
| 825 | 856 | 3 | Bitwise | Bitwise |
| 1745 | 1777 | 3 | Bitwise | Bitwise |

The production packed replay matches the saved live output bitwise in all four
snapshots. The legacy serial kernel, per-row packed kernel, and packed widths
1, 2, and 4 match on all 12 rows. Compared with the independently implemented
generic TurboQuant reader, the worst max absolute error is 0.125, worst RMSE
is 0.005925, worst relative L2 is 0.001147, and minimum elementwise equality is
86.36%. These are BF16-scale reduction-order differences, not bitwise agreement.

The synthetic boundary matrix in the earlier diagnosis remains complementary:
it covers more dtypes, block sizes, split counts, scales, and tile boundaries;
the snapshots establish that the same conclusions hold at selected points from
the real model and that snapshot extraction itself is byte preserving.

## Reproduction and artifacts

The correctness-only kernel harness now accepts real snapshots:

```bash
docker exec -e PYTHONPATH=/workspace/vllm -w /workspace/vllm \
  tq-e2e-current /tmp/tq-mtp-eval/.venv/bin/python \
  benchmarks/kernels/benchmark_turboquant_gfx1201_numerics.py \
  --output /tmp/real-qkv-replay.jsonl \
  --snapshots /tmp/tq-greedy-diag/multi-c-attn/attention-c*.pt
```

The harness does no timing and does not mutate the cache. The suite, raw A/C
responses, paired scores, sandboxed code-test results, acceptance metrics,
B/C hashes and batches, real-QKV snapshots, replay metrics, and a machine-readable
summary are preserved under `/tmp/tq-accuracy-eval.ReUkmn`. Exact launch and
evaluation instrumentation remains local under `.tools/tq_accuracy_eval/` and
`.tools/tq_greedy_diag/`.

Taken together with the prior synthetic, layerwise, target-only, RMSNorm, and
model-level controls, this moves the small model-accuracy gate to pass. It does
not require row-serial RMSNorm in production, claim bitwise invariance across
ordinary and speculative target shapes, or justify default enablement.
