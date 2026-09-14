# gfx1201 prefill rearchitecture

Status: plan and inactive interfaces only. No new performance result.

Branch: `feat/gfx1201-prefill-rearchitecture`

Base: `9938409e924ba0418b65dfae8e56304a070726a6`

This is a new implementation plan, not a reopening or relabeling of the failed
P2.2/P3/P3-v2/P4 candidates. Their code, artifacts, gates, and conclusions remain
unchanged. Stage completion is not acceptance of a production optimization.

## 1. Objective and preserved baseline

The objective is fast cold long-prompt processing and fast subsequent decode
on one R9700 in the **same usable configuration**, with qualified quality and
capacity. Importing a feature or passing a synthetic test is not the objective.

The user's historical Radiance measurements were approximately 2,100/1,735/1,326
prefill tokens/s at 32K/64K/120K, using TP1, MTP8, R4D, FP8 KV and a 131K envelope.
They are historical comparison targets, not matched-baseline acceptance numbers.
The user's llama.cpp decode experience is the other target; do not invent its
throughput. R0 records its exact model/quantization/runtime and measured rate
when available. Different weight quantizations are labeled, not quality-equated.

Retain the following at the base revision:

| Component | Action |
| --- | --- |
| gfx1201 K8/V4 SoA decode and MTP2 ragged verification | Preserve implementation and validated opt-in settings |
| Software-fused MXFP4 small-M BF16 linear | Preserve `triton_gfx1201.py`, M<=4/N>=512 lane and fallbacks |
| Compilation-disabled `FULL_DECODE_ONLY` | Preserve as the validated runtime lane; do not enable default O2 |
| KV/draft ownership, hybrid prefix reuse, workspace management | Preserve ownership, hash units, copy-on-write and graph address lifetimes |
| Failed attention/GEMM experiments | Keep as diagnostic references, never auto-enable or import into the new lane |
| Tests, snapshots, trace analyzers and artifacts | Reuse after checking their exact numerical and timing contracts |

K8/V4 remains the initial production cache. FP8 KV is allowed as a separate
control, not a silent replacement. No promise is made that both historical
performance targets can be attained with unchanged formats.

## 2. Evidence and its limits

Read these existing reports before implementation:

- [Validated rollout](turboquant_gfx1201_rollout_validation.md).
- [Decode graph qualification](turboquant_gfx1201_full_graph_validation.md).
- [MXFP4 fusion](turboquant_gfx1201_mxfp4_fusion.md).
- [Radiance controls](gfx1201_radiance_delta_diagnostic.md).
- [Closed v2 mapping and B-load experiments](gfx1201_p3_v2_radiance_w4a8_cleanroom.md).

The Radiance control changed TTFT from 24.825 s to 50.578 s with W4A8 disabled;
classified linear kernel time rose from 11.521 s to 37.214 s. This supports a
large W4A8 contribution **within that control**. It is not a matched comparison
to our MTP2 fork. Kernel-time sums are not wall-clock critical paths.

R4D-off retained an alternative attention backend. Its modest change does not
show that attention is unimportant relative to our Math-SDPA fallback. Fix
launch attribution for continuation Math GEMMs before using category shares.
Do not resurrect the superseded 75.334% linear classification.

The closed A3/B1/B2 study found a roughly 8.98x weighted latency gap between A3
pure FP8 and same-session BF16 GEMM at M=256. B1/B2 did not reduce that gap.
This motivates a coherent new operand/compute/output design, not more isolated
B-load order changes. A0 ratios and static WMMA opcode counts are diagnostics,
not estimates of available hardware performance.

### 2.1 Source exposure and provenance

Pinned Radiance image:
`magiccodingman/vllm-radiance@sha256:83a9dc02a8f8e75aabe81366d36ebaa2e35fcbe181cacf8e8e0a4cef4ebccbcc`.
Its recorded source label is `f295b9ef51ad413a68e4192371e0377741a354ce`.

The prior conversation subsequently inspected these public source files:

- [Radiance HIP source](https://github.com/magiccodingman/vllm-radiance/blob/f295b9ef51ad413a68e4192371e0377741a354ce/radiance_mxfp4_fp8.hip).
- [Radiance linear plugin](https://github.com/magiccodingman/vllm-radiance/blob/f295b9ef51ad413a68e4192371e0377741a354ce/radiance_mxfp4.py).
- [Radiance attention integration](https://github.com/magiccodingman/vllm-radiance/blob/f295b9ef51ad413a68e4192371e0377741a354ce/radiance_r4d_attn.py).

Therefore this plan is **source-informed**, not a claim of source-blind
clean-room development. The old black-box reports remain descriptions of their
original experiments. Do not rewrite that historical provenance.

R0 must inventory the exact source origins, licenses, notices and transitive
components before importing code, tables, build products or patches. A public
repository, a source label or independently typed code is not license clearance.
Where rights are unresolved, do not copy implementation text or lookup tables.
Use independently specified algorithms and verified AMD interfaces, record
source exposure, and flag reuse questions for review. Do not make a legal
assurance based on the phrase "clean-room".

AMD API reference for independently implemented matrix operations:
[rocWMMA API](https://rocmdocs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html).
Pin the local SDK/header revision in the manifest. Its cooperative-load API
and compiler scheduling barriers do not replace required workgroup barriers.
In particular, do not assume the cooperative API supports an eight-wave block.

### 2.2 Architecture hypothesis, not a copied implementation

The observed folded kernel name corresponds, in the pinned source, to a design
with load-time weight preparation, per-row reference exponents, wide staged
loads, K slabs, padded LDS, register accumulators and direct output stores.
Source inspection is not verification of every compiled binary instruction or
of each feature's performance contribution. Optional source features are not
assumed active in the measured image.

The new implementation must consider those components together. In particular:

1. Lossless byte permutation is separate from numerical scale folding.
2. Native FP8 GEMM efficiency must be established against same-session library
   GEMM, not only against our unusually slow A0.
3. An output-only LDS slab must not consume the block's shared-memory budget
   throughout the K loop. Keep results in registers and map final stores.
4. Use a K slab of at least 64 for the initial mapping, with multiple WMMA steps
   per staging/synchronization cycle. Do not rebuild the old K=16 load loop.
5. Specify lane ownership, vector alignment, padding, barriers and epilogue as
   one contract; verify it before evaluating performance.

## 3. Files and interfaces

This preparation commit adds inactive modules only. It does not modify imports,
CMake, op registration, environment variables, kernel selection or dispatch.

| Work ID | File/function to implement later | Connection boundary |
| --- | --- | --- |
| R0 | `benchmarks/kernels/benchmark_gfx1201_prefill_v3.py` | Frozen manifest, reference and measurement runners |
| R1 | `csrc/rocm/gfx1201_prefill_v3.cu` | Out-of-tree raw-FP8 GEMM and independent fragment mapping tests |
| R2 | `gfx1201_prefill_v3.py::prepare_weights` | Load-time prepared view; original weight remains untouched |
| R2 | `gfx1201_prefill_v3.py::quantize_activation` | Dynamic E4M3 producer into caller-owned buffers |
| R2/R3 | `gfx1201_prefill_v3.py::launch_prefill` | Prepared packed weight plus activation, explicit numerical mode |
| R4 | Future `MxFp4LinearKernel` adapter | Existing `linear/__init__.py::_POSSIBLE_MXFP4_KERNELS` only after qualification |
| R5 | Existing `TurboQuantAttentionImpl._continuation_prefill` | Replace attention consumer only after equivalent-input qualification |

Full Python interface path:
`vllm/model_executor/kernels/linear/mxfp4/gfx1201_prefill_v3.py`.
No class is registered by that module. Its eligibility function always returns
False and its launch/prepare methods raise `NotImplementedError`.

The future adapter delegates every ineligible call to the original validated
kernel instance. `EmulationMxfp4LinearKernel.apply_weights` is the full large-M
baseline: weight dequantization + activation QDQ + `F.linear` on every call.
Do not change that baseline to pre-expanded `torch.mm` and keep its old name.

## 4. Tensor, ownership and numerical contracts

### 4.1 Inputs and prepared state

Initial target profile: gfx1201, TP1, target text linears, BF16 input, bias absent,
N>=512, K divisible by 64. M=64 and 256 are separate measured buckets. N=96,
small-M decode, drafter, vision, unsupported dtypes/strides and mixed-role calls
retain the original path. Real phase identity is required: M alone does not
prove a call is prefill. R4 must resolve actual forward metadata without a GPU
synchronization; unknown phase means fallback.

Logical tensors:

- X: BF16 `[M,K]`; initial fast path requires contiguous last dimension and
  supported row stride.
- Original W: UINT8 `[N,K/2]`, low nibble at even K; original group scales:
  UINT8 `[N,K/32]`, OCP E8M0.
- Prepared W: optional UINT8 byte permutation, with a versioned layout tag;
  prepared scales have their explicitly documented layout, never inferred
  from the tensor's apparent shape.
- Aq: raw E4M3 bytes `[M,K]`; As: FP32 `[M]`; both produced into owned workspace.
- Y: BF16 `[M,N]` at the production boundary. Separate diagnostic output may
  expose the final pre-cast FP32 accumulator; do not time an instrumented kernel
  and label it production performance.

Prepared objects retain references to original tensors, not duplicate BF16
weights. If a second packed layout is retained for prefill, **count its full
persistent VRAM cost**. Do not describe it as metadata-only. Never hand permuted
bytes to checkpoint-layout fallback or to existing decode. No data_ptr-only
cache keys, in-place repacking, lazy preparation during graph capture, or
repacking on every token. The one-time cost and break-even usage are reported.

Workspace owns Aq, As, Y and explicit scratch per in-flight execution lane.
Caller lifetime must outlive asynchronous work. Do not share writable scratch
across concurrent streams without ordering; never allocate or resize during
capture. Chunk maxima, alignment and growth limits belong in the manifest.
No whole-model BF16/FP8 expanded-weight cache is assumed affordable on 32 GB.
The first production candidate must also work from canonical packed weights.
A second whole-model packed copy is not a default requirement: its additional
VRAM may destroy long-context capacity. A bounded per-layer preparation option
must charge its repeated cost; switching existing decode to a new weight
layout requires a separate reviewed, bitwise-validated change.

### 4.2 Separate three numerical changes

**Lossless layout:** pack/unpack round trip must preserve every original byte,
including tails and zero padding. The decode view is unchanged.

**Exact-weight W4A8:** activation changes from original Quark MXFP4 QDQ to
per-row E4M3 quantization; weight values retain their group-32 scale meaning.
Use R0's declared rounding/saturation/minimum-scale/NaN contract. Independent
reference consumes the actual generated E4M3 bytes, not a possibly different
Torch conversion. Compare activation conversion separately. NaNs are reported,
not silently zeroed to obtain a pass.

**Folded-weight W4A8:** for output channel n, let r[n] be the reference exponent
and e[n,g] the group exponent. Prepare/use FP8 values approximating
`E2M1[n,k] * 2**(e[n,g]-r[n])`, then restore `2**(r[n]-127)` and As in the
final epilogue. The reference exponent and each folded byte are independently
validated. The transformation can round/underflow small weights: it is not a
lossless layout change. Do not scale a running accumulator again at each group;
exact-group and folded accumulation require distinct references.

Handle E8M0 byte 0, byte 255, zero groups, exponent extremes and FP8 subnormals
explicitly according to the declared format contract. Do not transplant a
bit-shift shortcut valid only for normal exponents. Model snapshots must include
real group-exponent spreads, particularly down projections.

### 4.3 Reference hierarchy and gates

No P2.2 attention absolute threshold or all-shapes A0 5x requirement is reused.
R0 freezes the input set, numerical modes and machine-readable gate manifest
before a new candidate is measured. Later policy changes create a new experiment,
never overwrite the previous verdict.

For pure GEMM and exact-weight W4A8, retain FP64 pre-cast and FP64-to-BF16
references built from identical effective bytes and scales. Use independent
FP32 GEMM as a diagnostic, not a mathematical oracle. Suggested initial
engineering limits, explicitly not a model-quality theorem:

- all outputs finite for finite supported inputs;
- pre-cast normalized max error <=1e-3 and relative L2 <=1e-4 against FP64;
- final BF16 error against pre-cast FP64 <=1.10 times the BF16 cast-only error
  plus the above allowed pre-cast error (apply separately to max and L2 norms);
- exact zero-reference cases require exact zero, rather than dividing by a
  tiny arbitrary norm. Record cancellation-heavy cases independently.

Here normalized max is `max(abs(delta))/max(abs(reference))`; L2 uses the full
reference norm. The output allowance uses the reference's cast-only error, not
candidate-dependent tolerance. Also report BF16 mismatch fraction and ULP
histograms. Bitwise identity is mandatory for lossless reindexing, but is not a
universal requirement for a different GEMM or numerical format.

For folding, first check arithmetic against the **folded effective-weight**
FP64 oracle with the same rules. Separately measure the transformation's error
against unfurled group-scaled weights. Never hide folding error by using only
the transformed oracle. R3 must qualify that numerical change on model quality;
a single greedy hash, even with many downstream token differences, is neither
proof of a broken kernel nor proof of quality retention.

## 5. Execution stages and bounded decisions

### R0: freeze references and a usable comparison environment

Deliver a versioned manifest and reusable small input/snapshot inventory.
Reuse the coherent ROCm 7.14 development environment from the B1/B2 run; record
image digest, Torch/HIP/compiler/header versions and extension hash. Do not
change host drivers or upgrade the validated serving installation.

Record the accepted fork launch settings and all disabled experiment flags.
Fix the diagnostic flash override where necessary so the known working Math
fallback is actually selected; do not repeatedly launch a known crashing CK
kernel. Record the selected op, not a label inferred from an environment flag.

Capture exact effective `(M,N,K,dtype,stride,phase,role,calls)` before choosing
shape weights. Existing six-shape counts `(64,64,64,48,48,16)` are an initial
local reference, not proof of Radiance's distribution. An annotation q_len is
not automatically each GEMM's M. Do not compare a mixed-shape p50 to one GEMM.

Matched fork/Radiance controls use the same token-ID prompt, text-only startup
settings, chunk budget, no speculation for attribution, cold prefix cache,
warmed compilation and fixed outputs. Measure unprofiled TTFT separately from
profiled category sums. One successful preserved baseline plus a manifest is
sufficient here; no repeated 64K/120K runs or 308-case suite yet.

R0 completes only with provenance recorded and the baseline/shape/numerical
manifest frozen. Failure to build is `blocked_environment`, not kernel failure.
If a source cannot be reused, record it and use an independently specified route;
do not silently copy it or characterize mere code retyping as independence.

### R1: coherent raw-FP8 mapping, not another B1/B2 change

Implement benchmark-only native GEMM in the new CU file. Independently verify
lane-to-A/B/C mapping with basis-vector and row/column-distinct tests before
large matrices. Use official supported interfaces; eight-wave designs require
an explicitly tested mapping, not unsupported cooperative API assumptions.

Start with at most two coherent designs:

| Hypothesis | MxN tile | K slab | waves | Key requirement |
| --- | --- | --- | --- | --- |
| H1 | 128x64 | 64 | 4 | wide staging, independently checked padded LDS, direct output |
| H2 | 256x64 | 64 | 8 | explicit lane/register mapping and reuse across waves |

These are proposed source-informed experiments, not Radiance constants claimed
as original inventions. Do not copy its kernel or conversion table. Derive
mapping and any tables from the declared formats and verify them independently.

The design must include aligned 16-byte transfers where valid, safe zero-filled
edge handling, K=64 slab reuse and register accumulators. No full output-tile
LDS array. Preserve required barriers. Inspect compiler scheduling, private
memory, vector loads and actual LDS footprint. No global clamp trick is assumed
safe for arbitrary tails. Separate aligned fast path from safe tail handling.

Use same-session pre-expanded BF16 library GEMM with BF16 output, independent
FP32 control, old A3 as history, and H1/H2 with matching BF16 final output for
performance. FP32 debug output is timed and labeled separately. Also test an
already installed, documented native FP8 library path if it actually supports
gfx1201; unsupported is an explicit result, not a fallback labeled FP8.

The primary performance gate is prospective: at least **1.0x the same-session
BF16 GEMM** in call-weighted time on the five N>=512 shapes at M=256. This is a
minimum feasibility gate, not the final objective. M=64 is independently
qualified; it does not veto an M=256-only route. N=96 stays fallback. A0-relative
speedups do not decide this gate. A shape-specific choice between H1/H2 may be
selected on calibration data, but freeze that choice and rerun independent
validation samples before counting it as a result.

Use five warmups and at least twenty order-rotated samples, fixed buffers and
saved raw timing, with the existing 64 MiB flush condition explicitly labeled.
Repeat the selected mapping on a separate timing pass; report each shape and
weighted sums rather than averages of ratios. Existing traces/counters can
explain failures but do not substitute for measured speed.

At most one evidence-based revision per H1/H2 is allowed in this stage. If
neither reaches the gate, stop and report one architecture-level recommendation;
do not start another indefinite tile/load/scratch sweep or add MXFP4 work.
R1 success permits proposing R2; this task's first handoff stops at R1 results.

### R2: lossless packed layout and exact group-scaled W4A8

Implement `prepare_weights`, reversible layout transformation and independent
byte round-trip tests. Keep canonical weight and scale tensors for existing
decode/fallback. Freeze prepared-layout version and address formulas in tests.

Add dynamic E4M3 activation producer and exact group-scaled packed-weight GEMM
to the R1 mapping. Decode a packed tile once for the consumers that reuse it;
do not repeat nibble conversion independently for every output fragment.
Compute group contributions correctly; do not multiply prior groups by a new
scale. Both a group-local register partial and total accumulator may be needed.
The original one-wave kernel is a reference, not the new implementation body.

Measure quantization, GEMM and their combined call. Instrumented internal phase
timing is diagnostic only; the combined uninstrumented call decides speed.
Compare to production full emulation, including weight expansion and QDQ.
Charge any per-call permutation/copy. Report retained prepared-weight bytes,
workspace peaks, one-time prepare cost and original-weight ownership.

Proceed to model integration only after numerical gates and >=1.25x weighted
full-call speedup on the frozen eligible mix. Target >=2x full-emulation speedup
as an ambition review, not permission to change a failed gate retroactively.
R2 failure may lead to R3 only if the predeclared profile shows group scaling
is the limiting work and folded-input pure GEMM remains viable; otherwise stop.

### R3: optional scale folding, a separate numerical mode

Implement an independent scalar/FP64 transformation first. Exhaustively test
E2M1 codes and applicable exponent deltas, including subnormals and zero groups.
Do not copy Radiance lookup tables. Preserve an exact-group numerical mode and
record the local transformation loss per layer/channel/group.

Expose folding as explicit prepared-weight numerical metadata, not as a hidden
side effect of packing or an environment alias to an old candidate. Evaluate
actual model inputs and exponent distributions before benchmarking. Model
quality is required even when the kernel is perfect for the folded weights.
A failed folding mode does not invalidate an accepted exact-group mode.

### R4: opt-in target-prefill integration and model acceptance

Only qualified R2/R3 modes may connect via a new kernel plugin/custom-op
boundary. Preserve the original kernel delegate for all unsupported calls.
Resolve true target-prefill identity from existing forward metadata; unknown,
drafter, decode and graph capture calls take the delegate. Do not infer role
from layer-name substrings or change model-level prefix/draft ownership.

Implement process_weights_after_loading preparation, graph-safe workspace and
op fake/meta support before registration. Proposed independent opt-in:
`VLLM_ROCM_USE_GFX1201_PREFILL_V3=1`, default false. It is not added in this
preparation commit. Old failed flags never enable this path. A layout mismatch
is a fail-closed diagnostic, not reinterpretation of bytes.

Stage integration on 4K, then matched cold 32K/chunk256, then available larger
chunk buckets. For model speed, compare the same accepted fork with only the
new path changed. Add >=5 alternating cold, compile-warm runs per condition;
report median and variability. Require >=10% cold-32K TTFT reduction, no repeatable
>2% decode throughput regression, and no increase in errors/preemptions.
Also test decode from an identical prebuilt state to isolate runtime regressions
from altered prefill outputs/acceptance. MTP counters are diagnostics when
numerics change, not forced to match across a new numerical mode.

Reuse the frozen 308-case suite and scoring harness. Initial conservative
quality rule: no lower per-task score than the same-environment accepted
baseline, no new invalid/tool-format outputs in the fixed fixtures, and no
nonfinite state. This is sample-specific evidence, not population equivalence.
Ambiguous score differences remain unqualified; do not label a greedy hash as
a benchmark score. HumanEval execution stays in isolated restricted containers.

Retain the 3000-token/2976-prefix-hit probe and compare cold/warm outputs within
the candidate's own numerical mode. Then run the established serving soak.
No full qualification is run for a candidate that already fails speed.

### R5: independent continuation-attention backend work

R5 is a second track after R0, not contingent on linear adoption and not a
reason to restart P2.2. Start by testing an available, supported dense BF16
attention backend on identical materialized K/V, rather than writing another
scalar or per-query kernel. Runtime availability, provenance and numerical
agreement are prerequisites. A crashing CK path is not silently retried.

For initial integration retain the current dequant FP16-to-BF16 prefix contract
and raw current chunk. Replace only the attention consumer. For query row i,
visible keys satisfy `j <= cached_len + i`; do not assume a rectangular
`is_causal=True` API uses the same alignment. Test q1, q127/128/129, q256/512,
zero/nonzero prefix and representative long rows against explicit FP64 masks.
All-quantized direct-reader controls have a different current-chunk contract.

Measure full continuation cost, including materialization and actual temporary
allocation, plus cold-32K TTFT. Admission requires >=1.3x full-continuation
speedup at matched live shapes and >=10% cold-32K TTFT improvement, followed by
R4's quality/reuse checks. A backend that only accelerates the 0.008% first
chunk does not justify this work. Account for retained max-context workspace;
OOM recovery and timeouts are different outcomes.

If K8/V4 plus a supported dense consumer cannot meet the capacity/performance
target, propose a separate FP8-KV production branch for review. Do not change
both cache format and numerical backend in one experiment. R4D-off ablation
is not the same as our Math-SDPA fallback. No new GDN work without a measured
material cost fraction or a correctness requirement.

The 2026-09-14 R5 diagnostic loaded the pinned AMD Triton FlashAttention wheel
and observed `attn_fwd.kd`, but its fixed same-input numerical gate failed
(18/19 required cases). That evaluation stopped before timing and model
integration. Its strict numerical result remains unchanged. See
`gfx1201_prefill_rearchitecture_r5_20260914.md` and its machine-readable
artifact for the provenance, contract, and per-case results. This does not
reopen P2.2 or reject K8/V4 storage as a format.

A separate, user-authorized evaluation policy v2 preserves that strict result
but uses causal/input-contract checks, full-output finiteness and the documented
upstream-derived tolerance to permit timing. Its 32K-prefix/q256 continuation
speed gate passes. After the recorded host-RAM and bounded-container startup
OOMs, the separately authorized 16 GiB/swap-off retry completed cold32K.
Five alternating pairs measured median TTFT 131.796047 s baseline versus
60.643420 s candidate (2.1733x, 53.9869% reduction), passing the 10% speed gate.
See `gfx1201_r5_model_retry_16g_20260914.md` and commit `d5fec6675f` for the
unchanged result. The previous OOM and strict numerical diagnostics remain
in `gfx1201_prefill_rearchitecture_r5_speed_20260914.md` and the original
numerical report. Quality and operational qualification are now separately
authorized; production integration and default enablement are not.

The subsequent fixed 308-case paired MTP2 evaluation completed. Correct counts
were noninferior (GSM8K 36/36, MMLU 64/67, HumanEval 138/138, baseline/candidate),
but the existing syntax check found one new invalid HumanEval output
(158/157 syntax-valid). Qualification therefore stopped at quality; 32K
retention, decode, prefix reuse and soak were not run. The original HumanEval
judge also lacked Docker stdin attachment and returned false passes; those
invalid results are retained separately from corrected isolated execution.
See `gfx1201_r5_quality_operations_20260914.md`. Neither the TTFT pass nor the
strict Math diagnostic was overwritten, and no production path was adopted.

### R6: composition, capacity and final experience

Compose only accepted stages and preserve component toggles for rollback.
Report cold 32K/64K/120K TTFT and prefill rate, decode rate by context, total
latency, MTP acceptance, peak allocated/reserved/physical VRAM and failure counts.
Prefix reuse is separate from cold prefill. A baseline OOM has no speed ratio.
Stop a capacity sweep at the first error/timeout and report it without inventing
results for longer contexts.

Compare qualified MTP2 first. MTP8 is a separately labeled historical control.
Compare llama.cpp using its actual pinned model and settings; do not multiply
speedups measured in different environments. A useful partial result can be
retained, but call the ultimate goal unmet until both prefill and decode targets
are demonstrated in one quality-qualified lane.

## 6. Commit sequence and first assignment

| Commit | Allowed content | Exit |
| --- | --- | --- |
| Plan (this commit) | Documentation, inactive interfaces, standalone tests | Import/syntax checks; no runtime behavior |
| R0 | Provenance/build/shape/numerical manifest and reusable references | Complete comparable baseline; exact gate rules fixed |
| R1 | New out-of-tree raw-FP8 mapping + benchmarks/tests | Same-session library-relative speed and correctness; then stop/report |
| R2 | Reversible prepared weights and exact W4A8 | Full-call numerical/speed/VRAM gates |
| R3, optional | Explicit folding mode | Separate transformation and model-quality qualification |
| R4 | Qualified opt-in integration only | Cold32K, decode, quality, prefix and soak |
| R5, independent | Qualified continuation backend only | Matched numerical/capacity/speed gates |
| R6 | Composition report | Same-configuration final experience |

Luna's first assignment is **R0 through R1, then report**. It is not authorized
to skip the library baseline, quietly enable old flags, enter R2 automatically,
or rewrite old failures as success. When a gate fails, report the smallest
supported conclusion, not that FP8/RDNA4/K8V4 as a whole is impossible.

Do not delete an existing `.gitignore` edit, reset another worktree, clean
untracked `.tools/`, or remove prior artifacts. Use a new worktree for this
branch when the original workspace contains user changes.

Preparation checks are documented in the handoff. GPU, full repository hooks,
ROCm compilation and performance remain for the implementation environment.
