# gfx1201 TurboQuant MTP rollout validation

Date: 2026-09-12. Source: `253fe700a9`. Hardware: Radeon AI PRO R9700,
gfx1201, ROCm 7.2. Native-MXFP4 hardware evaluation is intentionally outside
the scope of this rollout decision.

## Result

The remaining prefix-cache, broader-quality, and sustained-serving gates pass
for the opt-in gfx1201 TurboQuant MTP profile. The implementation now identifies
draft KV ownership from the loaded model instead of layer-name conventions, and
hybrid full-attention/Mamba groups can publish and replay a shared fine-grained
prefix boundary. A live repeated 3000-token request reuses 2976 tokens and
returns exactly the same 32 output tokens as the first request.

A 308-case paired quality run shows no aggregate task regression against the
target-only control. A 24-minute, four-request serving soak completes all 484
requests and 30,976 generated tokens without an error, abort, or preemption.

Keep both `VLLM_TQ_GFX1201_K8V4` and
`VLLM_ROCM_USE_GFX1201_MXFP4_GEMM` opt-in. This is a default-policy decision,
not a failed MTP validation gate: the repository's default O2 path was already
measured 61--69% slower than compilation-disabled `FULL_DECODE_ONLY`, and its
deterministic outputs differ from that path. The validated rollout profile is
therefore explicit opt-in with compilation disabled and full-decode graphs;
default O2 needs a separate compiler-path investigation.

## Prefix-cache ownership and replay

The previous warning and disabled-cache behavior came from treating a generic
MTP attention spec as indistinguishable from target attention. Model runners
now mark attention layers owned by the loaded drafter. The marker is scheduler
metadata: it does not participate in KV-spec equality, hashing, layout, or
grouping. Existing DSpark and DeepSeek-V4 detection remain fallbacks.

The live model exposes another hybrid-cache constraint. Its configured hash
unit is 16 tokens, while cache sizing resolves the Mamba scheduler group to a
2096-token block. The engine now preserves the configured unit before executor
initialization can replace `cache_config.block_size`, provided every
prefix-cacheable group is divisible by that unit. Scheduler chunk splitting
uses the resolved Mamba block rather than the hash unit.

For a 3000-token prompt, the safe common replay boundary is 2976. Full-attention
groups index that hash-aligned boundary as an alias, and the Mamba manager may
publish its partial state there even when the model has no internal Mamba
checkpoint slots. The existing copy-on-write path protects that state while
the first request continues through the remaining 24 prompt tokens.

The live two-request probe records:

| Measurement | First request | Repeated request |
| --- | ---: | ---: |
| Prompt tokens | 3000 | 3000 |
| Prefix-cache hits | 0 | 2976 |
| End-to-end latency | 6.487 s | 3.045 s |
| Output tokens | 32 | 32 |
| Draft tokens accepted | 44/44 | 44/44 |

Both requests return the same token IDs. There are no preemptions, and startup
no longer emits the draft-group-identification warning. The latency is a probe,
not a controlled throughput benchmark; the functional result is the 2976-token
cross-request hit shared by all participating cache groups.

## Broader quality evaluation

The expanded deterministic suite uses seed 1201 and contains 64 GSM8K cases,
all 164 HumanEval cases, and 80 MMLU cases (16 from each of the same five
subjects as the small gate). It spans 79--1719 prompt tokens and 97,001 total
prompt tokens. The suite SHA-256 is
`8476e86f75bc2b08a19f187e6ca11bef4af161bed203a0817c39a50b1150338a`.

Target-only (A) uses the ordinary non-speculative route. MTP (C) uses two draft
tokens, adaptive verification disabled, eager execution, and four concurrent
requests. Sampling and datasets otherwise match.

| Task | Cases | A | C | Semantic answers equal |
| --- | ---: | ---: | ---: | ---: |
| GSM8K exact answer | 64 | 34 | 41 | 37 |
| HumanEval functional tests | 164 | 164 | 164 | 164 |
| MMLU exact choice | 80 | 64 | 66 | 76 |

HumanEval candidates are executed in disposable containers with networking
disabled, a read-only root filesystem, no Linux capabilities,
`no-new-privileges`, and resource limits. All 164 candidates pass in both
modes. MTP completes all 308 requests without client errors or preemptions and
emits 22,803 tokens. The score differences favor MTP in this sample, but they
should be read as evidence against a broad regression rather than evidence that
speculation improves model quality; unlike execution shapes need not produce
identical greedy continuations.

## Sustained serving

The soak uses one warmup and 120 measured rounds. Each synchronized round sends
four requests with 128, 512, 1024, and 3072 prompt tokens. Every request ignores
EOS and emits exactly 64 tokens. The request-time envelope is 1442.16 seconds
(24.0 minutes), including warmup.

| Prompt tokens | Requests | Median decode tok/s | Median E2E tok/s | Token hashes |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 121 | 9.627 | 5.469 | 2 |
| 512 | 121 | 9.624 | 5.470 | 1 |
| 1024 | 121 | 9.630 | 5.471 | 1 |
| 3072 | 121 | 9.509 | 5.461 | 2 |

All 484 requests finish by length; error, abort, and preemption counters remain
zero. The server records 12,177 speculative steps, 24,354 proposed draft
tokens, and 18,811 accepted draft tokens. The two hashes at 128 and 3072 are
rare batch-shape-dependent greedy variations already permitted by the quality
criterion; completion, serving stability, and acceptance remain intact.

## Verification and artifacts

The focused CPU suite covers draft ownership, group annotation, fine-grained
hybrid replay, no-checkpoint Mamba replay, resolved Mamba chunk alignment, and
block-size resolution: 222 tests pass. Pre-commit passes on every changed
production and test file.

Raw quality responses, paired scores, isolated HumanEval results, soak records,
Prometheus snapshots, and the suite manifest are preserved under
`/tmp/tq-final-validation.Awe2nr`. Local launch and evaluation helpers remain
under `.tools/tq_accuracy_eval/` and are not part of the production change.

Together with the prior full-decode graph, multi-request, 32K-context, kernel
numerics, and small quality results, this closes the requested MTP validation
work. It does not change production defaults or claim results for native-MXFP4
hardware.
