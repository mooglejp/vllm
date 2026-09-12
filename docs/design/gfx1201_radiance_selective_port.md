# gfx1201 Radiance selective-port implementation plan

Status: phase 3 decode prototype complete; A3 speed gate failed; production
adoption stopped

Base revision: `ef9433f1156ab44d8b2445778b83011605a9892f`

Target branch: `feat/gfx1201-radiance-port`

Primary external reference: `magiccodingman/vllm-radiance` at
`adf9e1f1c9529dd6c971b223a961833376dbd524`.

The first commit on this branch must not change runtime dispatch, environment
variables, model outputs, cache layout, compilation policy, or default behavior.
The code scaffolds named below are deliberately not imported or registered.

## 1. Baseline that must remain available

The base revision closes the requested gfx1201 TurboQuant/MTP rollout gates for
the currently validated profile:

- Radeon AI PRO R9700 / gfx1201 / ROCm;
- Quark MXFP4 target weights with the existing software-fused gfx1201 MXFP4
  decode linear;
- TurboQuant K8/V4 cache and the specialized single-/multi-token attention path;
- two-token Qwen3.5 MTP with target verification;
- hybrid FullAttention/Mamba prefix reuse, including the validated 2976-token hit
  on a repeated 3000-token request;
- compilation disabled with full-decode graphs as the validated serving policy;
- all current opt-ins remain opt-in.

This baseline is the rollback target for every phase below. Do not delete or
replace `TritonGfx1201Mxfp4LinearKernel`, the TurboQuant attention path, the
existing Qwen3.5 MTP implementation, or the hybrid-prefix-cache fixes while
prototyping Radiance-derived ideas.

## 2. Source and provenance gate (R0)

Before copying implementation code from any external project, record the exact
source commit, file, author/provenance chain, and license that permits copying
into this Apache-2.0 vLLM fork.

The GitHub mirror of `magiccodingman/vllm-radiance` currently exposes no
repository license through GitHub metadata. Its README attributes major pieces
to StillDeadcode/libr4d and ggz14/Radiance MXFP4 work. Therefore:

1. do not copy external implementation code until the relevant source license is
   identified and compatible;
2. preserve SPDX/copyright notices required by the source;
3. if licensing is unclear, reimplement from the documented algorithm and public
   behavior rather than transliterating source code;
4. record the decision in this document before the first functional port commit.

Phase 2 provenance decision (2026-09-12): the pinned GitHub mirror does not
expose a repository license, and the README attributes major pieces to other
projects. No Radiance implementation code is copied into vLLM. The phase 2
W4A8 reference is a clean-room tensor implementation of the public A2 contract
in this document, with the external commit retained only as a provenance and
comparison pin. No external copyright notice is required for the reference
implementation beyond the vLLM Apache-2.0 headers.

Reference material for the first two workstreams:

- `radiance_mxfp4.py` and `radiance_mxfp4_fp8.hip`: packed MXFP4 + dynamic FP8
  activation W4A8 path, decode/prefill separation, optional fragment-order
  weights and non-temporal decode loads;
- `docs/MXFP4_W4A8_R9700.md` and
  `docs/MXFP4_RX5_FP8KV_CONTINUATION.md`: qualification and negative results;
- `radiance_drafthead.py`: draft-only INT2-g128 coarse head with BF16 exact
  reranking;
- Radiance `main` pin above; do not follow floating `main` during an experiment.

## 3. Candidate priority

| Priority | Candidate | Why now | Numerical class |
| --- | --- | --- | --- |
| P0 | gfx1201 MXFP4/W4A8 target linear | Current long-context E2E remains prefill-limited; current backend only fuses small-M software emulation | Changes target arithmetic |
| P0 | draft-only INT2 exact-rerank head | Full BF16 vocabulary projection is bandwidth-bound; reducing draft-head bytes attacks the right resource instead of retuning `wvSplitK` | Changes proposals, target verification unchanged |
| P1 | W4A8 fragment-order weight permutation + decode non-temporal loads | Radiance reports a safe subset gain, but only after the base W4A8 contract is proven | Should not change W4A8 math, but changes stored layout |
| P2 | W4A8 large-M/prefill tiling | 3072-token + 64-output workloads spend the majority of kernel time in prefill | Changes target arithmetic |
| P3 | DFlash2 / dynamic speculation / TP2 collectives | Separate serving mode and validation problem | Separate project |

Do not spend the first port cycle on R4D attention, GDN state kernels, or a
replacement BF16 vocabulary GEMM. The local profile already puts target
attention and GDN state/conv around the low-single-digit share, and the existing
`wvSplitK` full-vocabulary projection is already near the R9700 bandwidth roof.

## 4. Global implementation rules for Luna

These rules are mandatory for every phase:

- one hypothesis per commit; do not combine W4A8, draft-head approximation,
  attention, GDN, compilation-policy, or prefix-cache changes;
- all new runtime paths start behind a new explicit opt-in and preserve the
  previous backend as a per-call fallback;
- do not change current defaults, `VLLM_TQ_GFX1201_K8V4`,
  `VLLM_ROCM_USE_GFX1201_MXFP4_GEMM`, compilation policy, MTP depth, cache
  layout, or scheduler policy unless the phase explicitly says to do so;
- unsupported architecture, dtype, shape, bias, quantization recipe, or missing
  extension must fail closed to the existing implementation;
- never reinterpret an already-written weight/cache layout with the wrong
  reader; any weight permutation must be selected once after load and paired
  with only compatible kernels;
- record generated ISA metadata (VGPR, SGPR, LDS/shared, scratch/private segment,
  wave size) for every adopted gfx1201 HIP/Triton kernel;
- benchmark with preallocated outputs for kernel-only comparisons, rotating A/B
  order and saving raw samples;
- use same-process/same-model-load comparisons for sub-5% model-level claims;
- do not infer model quality from token-hash equality alone when arithmetic is
  intentionally changed;
- stop the phase when its gate fails. Do not broaden the optimization search to
  unrelated kernels without updating this design first.

## 5. Workstream A: gfx1201 MXFP4/W4A8 target linear

### A0. Keep the existing backend intact

Current connection point:

- `vllm/model_executor/kernels/linear/mxfp4/triton_gfx1201.py`
- selected through the existing MXFP4 plugin list in
  `vllm/model_executor/kernels/linear/__init__.py`.

New implementation goes in:

- Python/backend contract:
  `vllm/model_executor/kernels/linear/mxfp4/gfx1201_w4a8.py`;
- ROCm/HIP kernel source when the prototype reaches native WMMA:
  `csrc/quantization/gfx1201/mxfp4_w4a8.hip`;
- declaration: `csrc/ops.h`;
- ROCm registration: `csrc/torch_bindings.cpp`;
- build source list: append only the new HIP source to the ROCm legacy `_C`
  target in `CMakeLists.txt` unless the implementation is deliberately moved to
  `_C_stable_libtorch` with its own design update.

Do not put W4A8 code into `triton_gfx1201.py`. W4A8 changes activation
quantization semantics and must remain a separately selectable backend.

### A1. Proposed public gate and backend selection

When implementation begins, add an environment setting named
`VLLM_ROCM_USE_GFX1201_MXFP4_W4A8`, default `False`.

Add a class named `Gfx1201Mxfp4W4A8LinearKernel(MxFp4LinearKernel)` and insert it
in ROCm MXFP4 priority order after any actually supported native-MX backend and
before `TritonGfx1201Mxfp4LinearKernel` / emulation.

`is_supported()` must require all of:

- explicit opt-in;
- ROCm gfx1201;
- required HIP custom op present;
- Quark package present only if the chosen activation-quant reference still
  depends on Quark.

`can_implement()` must initially require dynamic MXFP4 activation metadata. The
backend receives BF16 model activations but deliberately quantizes them to FP8
E4M3 for the W4A8 GEMM. This is a numerical-mode change, not an exact
implementation of the current W4A4 recipe.

Per-call fallback to the existing `TritonGfx1201Mxfp4LinearKernel` or emulation
must remain available for unsupported rows/shapes/bias/dtypes.

### A2. Weight and activation contracts

Checkpoint weight input remains OCP MXFP4:

- packed E2M1 values, two values per byte;
- group size 32 along K;
- E8M0 scale byte per group;
- logical weight shape `[N, K]`, packed storage `[N, K/2]`.

Initial functional port must use checkpoint row-major packed order. Do not add
weight permutation in the same commit.

Activation contract for the W4A8 experiment:

- source activation: BF16 contiguous `[M, K]`;
- quantized activation: FP8 E4M3 with explicit per-row scale;
- scale computation must have a Python/reference implementation used in tests;
- output: BF16 `[M, N]`;
- accumulation: FP32 unless a separately validated HIP instruction contract
  proves another behavior;
- bias: unsupported in the first port; fallback if present.

The first reference implementation must be independent of the HIP kernel:
quantize/dequantize the FP8 activation with the exact chosen scale rule,
dequantize MXFP4 weights, run `F.linear`, and compare with the candidate.

### A3. Decode prototype before prefill

Start with the shapes already observed locally, especially M=1 and M=3. Do not
port the whole Radiance M<=64/prefill stack at once.

Required first matrix:

- all five eligible Qwen3.5-27B dense `(N,K)` shapes already tracked by
  `benchmark_mxfp4_gfx1201.py`;
- M in `{1,2,3,4}`;
- real captured QDQ/source activations from the model in addition to seeded
  tensors;
- masked N/K-tail unit cases.

Adoption gate for decode W4A8:

1. no NaN/Inf outside values expected from an explicit NaN input/scale case;
2. candidate matches its independent W4A8 reference within a measured and
   documented numerical envelope on every shape; do not reuse the old W4A4
   bitwise gate;
3. production-call-weighted kernel time is at least 1.10x faster than the
   current software-fused MXFP4 backend for M=3, or the phase stops;
4. same-load model decode improves by at least 3% before claiming an E2E win;
5. run the 308-case paired quality gate before adoption. HumanEval must remain
   164/164, and neither GSM8K nor MMLU aggregate may be lower than the validated
   baseline without an explicit review decision;
6. run the existing 12-prompt/64-position instrumentation and record logits,
   token traces, MTP acceptance, and input batches. Bitwise equality to W4A4 is
   not required because W4A8 is intentionally different.

If step 3 fails, retain the scaffold and stop. Do not compensate by enabling
other Radiance features in the same commit.

### A4. Fragment-order weights and non-temporal decode loads

Only after A3 passes.

Implement weight permutation as a one-time post-load transform with an explicit
layout enum/state. The transformed tensor must preserve the original logical
shape metadata or otherwise update every consumer deliberately. Keep the
original row-major representation if any fallback path still requires it.

Requirements:

- round-trip permutation test on synthetic packed bytes;
- direct equality of the decoded logical MXFP4 matrix before/after permutation;
- no fallback may read fragment-order bytes as checkpoint-order bytes;
- WPERM and the compatible reader must be enabled by the same state, not
  independent flags;
- non-temporal load behavior must be a kernel implementation detail under the
  same numerical contract.

Adopt only if the safe subset improves same-load decode by at least 2% or the
weighted W4A8 kernel by at least 5% with no quality regression.

### A5. Prefill/large-M W4A8

Only after decode W4A8 is validated.

Build a separate large-M launch policy rather than stretching the small-M
kernel. Benchmark at minimum M `{16,64,128,512,1024,2048,3072,4096}` for the
real model shapes that execute during prefill.

Keep the current software/emulation path as fallback for unsupported M.

Long-context gate:

- candidate prefill GEMM must beat the existing path by at least 1.25x on a
  production-call-weighted 1K--4K mix;
- 3072-token + fixed-output E2E must improve by at least 5% in same-load A/B;
- repeat the 308-case quality gate and prefix-cache replay probe;
- compilation-disabled `FULL_DECODE_ONLY` remains the serving policy during
  this port. Do not mix O2 compiler diagnosis into A5.

## 6. Workstream B: draft-only INT2 exact-rerank head

### B0. Integration boundary

Current Qwen3.5 MTP model:

- `Qwen3_5MTP.compute_logits()` in
  `vllm/model_executor/models/qwen3_5_mtp.py` calls its own
  `LogitsProcessor` with the shared `ParallelLMHead`;
- the target uses a separate logits processor even when the BF16 lm-head tensor
  is shared.

The optimization must attach only to the drafter's proposal path. Do not mutate
or replace the target `LogitsProcessor`, and do not remove the BF16 shared head:
it remains the target verifier's source of truth and the exact rerank weight.

New helper module:

- `vllm/model_executor/layers/gfx1201_fast_draft_head.py`.

When functional implementation begins, add an explicit opt-in named
`VLLM_ROCM_USE_GFX1201_FAST_DRAFT_HEAD`, default `False`.

Do not monkeypatch class methods globally. Prefer an explicit model-owned helper
or logits-processor strategy selected in `Qwen3_5MTP.__init__` / load-finalize
logic, so target and drafter ownership is obvious from the object graph.

### B1. Initial algorithm contract

Start from the Radiance algorithmic choices, but implement under the provenance
rule in R0:

- auxiliary draft weight: asymmetric INT2, group size 128 along K;
- quarter-split packing for the coarse kernel;
- coarse output block width 64;
- `KCAND=8` coarse candidates per block as the initial value;
- sweep exact rerank width `{32,64}` rather than assuming one setting;
- exact rerank uses the original BF16 head rows for selected candidates;
- BF16 target head stays resident and shared;
- auxiliary packed state is keyed/owned by the actual weight object or an
  explicit model field, never by a module-name guess.

The first implementation target is MTP argmax proposal only. Do not generalize
to DFlash top-k in the same phase.

### B2. Candidate-recall gate before serving

Collect at least 8192 real MTP draft hidden-state rows from the validated model
workload, spanning short, medium, long, code, math, tool/structured, and
reasoning prompts.

For each row, compare the fast draft result with the exact existing BF16
`wvSplitK` argmax.

Required gate before model integration:

- exact argmax recall 100% on the captured set;
- no invalid candidate IDs;
- rerank always includes the returned winner;
- packed-state creation does not create an FP32 full-head temporary and does not
  leave a large transient allocator reservation;
- record auxiliary VRAM size.

If 100% recall fails, first sweep `KCAND` and rerank width within the bounded
matrix documented in the benchmark. Do not silently accept misses because the
target verifier exists; a proposal-quality regression can erase the speedup.

### B3. Serving adoption gate

Run same-load A/B with identical target backend and MTP depth.

Adopt only if all are true:

- drafter vocabulary-projection time drops at least 2x;
- total decode throughput improves at least 5%;
- mean accepted tokens/verification step does not drop by more than 1% relative;
- 308-case task gate has no aggregate regression by the same policy as A3;
- 24-minute/484-request soak completes without request error, abort, preemption,
  invalid token, or memory growth;
- target verification remains untouched in code review.

A different draft proposal sequence is allowed if the gate above passes; do not
require the previous 768 full-target-logit hashes to remain bitwise identical
across unlike proposal/execution shapes. Instead keep the existing target-only
B-vs-C instrumentation available to prove the draft model does not alter target
arithmetic when replaying identical proposals.

## 7. Deferred Radiance ideas

Do not implement these until A and B are finished and a new whole-model profile
justifies them:

- R4D attention replacement: current TurboQuant attention is already a small
  share locally and has independent K8/V4 cache semantics;
- GDN fused/state kernels: current local state/conv share is small; revisit only
  after W4A8 changes the profile;
- DFlash2: separate drafter model, acceptance, memory, graph, and quality study;
- dynamic speculative depth: evaluate only after the fixed MTP fast-head path;
- TP2 custom all-reduce: current target is TP1 and the Radiance measurements are
  largely TP2-specific;
- FP8 KV calibration: fidelity/capacity experiment, not needed for K8/V4 cache;
- full RX4/RX5 norm-quant and FP8 residual-stream bundle: Radiance itself keeps
  these disabled after structured-tool regressions;
- O2 compiler/default-policy work: separate branch; the validated base remains
  compilation-disabled `FULL_DECODE_ONLY`.

## 8. Required benchmark/test files during implementation

Luna should create or extend these exact assets as the phases land:

- `benchmarks/kernels/benchmark_gfx1201_w4a8.py`
    - direct preallocated-output kernel timing;
    - current software-fused MXFP4 reference;
    - independent W4A8 numerical reference;
    - M/shape sweep and raw samples;
- `tests/kernels/quantization/test_mxfp4_gfx1201_w4a8.py`
    - packing, E2M1/E8M0 boundaries, FP8-scale boundaries, N/K tails, all real
    dense shapes, M=1..4 initially;
- `benchmarks/kernels/benchmark_gfx1201_fast_draft_head.py`
    - exact BF16 argmax, INT2 coarse candidates, exact rerank, recall and timing;
- `tests/kernels/test_gfx1201_fast_draft_head.py`
    - packing round trip/reference, candidate bounds, rerank correctness, unsupported
    fallback;
- reuse the existing rollout-quality/soak instrumentation rather than creating
  a second incompatible model-level gate.

Every benchmark result adopted into production gets a design report under
`docs/design/` with source pin, environment, hardware, exact command, raw artifact
location, negative candidates, and rollback instructions.

## 9. Planned commit sequence

Phase 2 gate record (2026-09-12): the independent BF16-to-FP8 E4M3 activation
reference, row-major MXFP4 E2M1/E8M0 decoder, and BF16-output W4A8 linear
reference are covered by CPU tests for row scales, zero rows, finite FP8
clamping, nibble order, E8M0 edge values, shape validation, M=1..4, and
non-registration. The phase-2 test gate passes on the target environment. No
production import, dispatch, environment variable, model integration, cache
layout, or default behavior is changed. Phase 3 remains a separate decode-only
step and must satisfy the A3 adoption gate before any later phase is started.

Phase 3 gate record (2026-09-12): the opt-in row-major Triton W4A8 decode
prototype was compiled and checked on a Radeon AI PRO R9700 (gfx1201) in the
`tq-e2e-current` ROCm container (`torch 2.12.0+git6bbd260`, HIP
`7.2.53211`). The direct kernel uses preallocated outputs and compares against
the independent reference bitwise for M in `{1, 2, 3, 4}` and `(N,K)` in
`{(5,64), (67,128), (65,160)}`; all
cases were finite and exact. The six production dense shapes were then timed
at M=3 with 5 warmups, 30 round-robin samples, a 64 MiB cache flush, and
`BLOCK_M=16, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=1`:

| N | K | W4A8 median (us) | MXFP4 median (us) | MXFP4/W4A8 |
| ---: | ---: | ---: | ---: | ---: |
| 5120 | 6144 | 147.50 | 106.64 | 0.723 |
| 34816 | 5120 | 496.02 | 377.06 | 0.760 |
| 5120 | 17408 | 365.84 | 270.82 | 0.740 |
| 16384 | 5120 | 246.12 | 198.46 | 0.806 |
| 96 | 5120 | 78.82 | 51.60 | 0.655 |
| 14336 | 5120 | 218.68 | 188.78 | 0.863 |

The six-shape unweighted geometric mean was 0.755x and the aggregate median
latency ratio was 0.768x, so the required 1.10x production-call-weighted
kernel speed gate failed. The raw JSONL artifact is `/tmp/tq-w4a8-a3.jsonl`;
the reproducible command was:

```text
docker exec tq-e2e-current bash -lc 'cd /workspace/vllm && /tmp/tq-venv/bin/python benchmarks/kernels/benchmark_gfx1201_w4a8.py --output /tmp/tq-w4a8-a3.jsonl --rows 3 --warmups 5 --samples 30'
```

The prototype remains behind the new default-off
`VLLM_ROCM_USE_GFX1201_MXFP4_W4A8` opt-in and falls back per call for
unsupported inputs. Because the A3 speed gate failed, no model-quality gate,
same-load E2E claim, fragment-order/non-temporal-load work (A4), or prefill
work (A5) was started. Disable the opt-in, or remove the prototype
registration, to return to the existing MXFP4 backend.

Use this sequence unless a phase fails its gate:

1. `[gfx1201] Plan selective Radiance ports` — this document + inert scaffolds.
2. `[MXFP4] Add gfx1201 W4A8 reference and tests` — no production dispatch.
3. `[MXFP4] Add opt-in gfx1201 W4A8 decode backend` — decode only.
4. `[MXFP4] Evaluate fragment-order/NT decode loads` — adopt or document no-op.
5. `[MXFP4] Add opt-in gfx1201 W4A8 prefill` — only if A3 passed.
6. `[MTP] Add gfx1201 draft-head reference and capture harness` — no dispatch.
7. `[MTP] Add opt-in INT2 exact-rerank draft head` — Qwen3.5 MTP only.
8. `[gfx1201] Re-profile selective Radiance subset` — decide whether any P2/P3
   work is justified.

A failed experiment should still commit its benchmark/report when the negative
result is informative, but production code for an unadopted candidate should be
removed or remain unreachable behind an explicit experimental gate.

## 10. Definition of done for the selective-port cycle

The cycle is complete when:

- the base `ef9433f115` behavior remains available by disabling new opt-ins;
- every adopted external idea has source/provenance/license notes;
- every adopted numerical-mode change has an independent kernel reference and a
  model-quality gate;
- W4A8 decode/prefill and draft-head fast path are each independently reversible;
- prefix reuse, full-decode graph replay, multi-request serving, 32K context,
  task quality, and sustained serving still pass for the final selected subset;
- a final whole-model profile names the next bottleneck instead of assuming the
  next Radiance feature is useful locally.

Do not enable any new path by default as part of this cycle. Default enablement
is a separate decision after compiler-policy work and broader deployment
qualification.
