# TurboQuant gfx1201 K8/V4 D=256 GQA=6 fused decode + MTP implementation plan

Status: implementation plan; opt-in single- and multi-token path implemented

The target route is currently gated by ``VLLM_TQ_GFX1201_K8V4``. Numerical
coverage and backend-level HIP graph replay validation are in place.
Initial eager-mode MTP acceptance/performance measurements are available below.
Output-equivalence investigation, wider-context and full-model graph
measurements, single-token tuning, and default-route replacement remain open
exit gates.

Validation follow-up (2026-09-11, gfx1201 / ROCm 7.2):

- TurboQuant explicitly opts out of device/CPU query-length mismatch because
  non-target speculative queries still use CPU-planned prefill paths. Adaptive
  verification remains unsupported by the backend.
- The target decode grid uses the configured speculative query-length bound.
  Packed stage 2 skips device-side padding before looking up the request.
  Host validation allows trailing padding and never reads device `seq_lens`.
- GPU regressions cover FP16/BF16, block sizes 16/32, ragged query lengths,
  empty requests, untouched padded output/LSE rows, and mixed MTP decode plus
  raw-KV prefill. A BF16/block-16 backend graph with query bound 5 replays with
  changed query/context lengths and all-padding input using a locked workspace
  with unchanged base pointers. This is not a full-model serving graph test.
- `tests/quantization/test_turboquant.py`: 169 passed, 2 skipped using
  `/tmp/tq-mtp-fixes/.venv/bin/python -m pytest tests/quantization/test_turboquant.py -q`
  inside the prepared `tq-rocm-ab` container. The isolated environment was
  created with `uv venv --system-site-packages --python /usr/bin/python`.

Workspace capacity is capped by `max_num_batched_tokens`; capture sizes already
count tokens and are not multiplied by the speculative query bound again.
Measured persistent GPU buffer sizes for Hq=24, D=256, split count 32, BF16,
256 maximum requests and query bound 5 are:

| Scheduler token budget | Largest capture (tokens) | Reserved decode tokens | GPU workspace |
| --- | --- | --- | --- |
| 256 | 256 | 256 | 195.8 MiB |
| 512 | 512 | 512 | 391.5 MiB |
| 2048 | 512 | 1280 | 978.9 MiB |

Builder reservation and implementation prewarm allocate the same exact aligned
shapes; all three measurements retained the same buffer size and pointer after
prewarm. A configuration that can actually schedule 1280 verification tokens
still needs nearly 1 GiB of split-K workspace.

Quark checkpoint loading follow-up (2026-09-11):

- The local `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` checkpoint stores all 15 MTP
  tensors in BF16. Its Quark exclusion list names parameters such as
  `mtp.layers.0.self_attn.q_proj.weight`, while vLLM checks module names.
  Literal `.weight` exclusions are now normalized before fused-layer matching;
  regexes retain their existing meaning and partial fused exclusions still fail.
- With the checkpoint unchanged, MTP FC/QKV/gate-up dispatch to
  `UnquantizedLinearMethod`, while the main decoder retains Quark MXFP4.
  The real model now loads and generates successfully with two MTP draft tokens.
- Configuration utility tests: 26 passed. Five new regression cases fail without
  the fix. The separate `test_quark.py` suite could not be collected in the test
  environment because the optional `lm_eval` dependency was absent.

Real-model evaluation uses ROCm 7.2 on gfx1201, TP=1, the V2 model runner,
`turboquant_k8v4`, `VLLM_TQ_GFX1201_K8V4=true`, requested block size 16, max model
length and scheduler token budget 4096, and four maximum requests. Hybrid cache
alignment produces physical pages of 2096 tokens with MTP versus 2080 without
MTP, resolving to kernel blocks of 16 and 32 respectively. RunAI loading uses
CPU staging (`distributed=false`, `memory_limit=3221225472`) to avoid the
draft-loader's extra GPU clone. This is an eager-mode run with Quark's MXFP4
emulation, forced SDPA prefill, and generic kernel warmup disabled; per-shape
inference warmups are excluded from measurements. It is not comparable to the
earlier 29.5/17.7 tok/s baseline from a different serving runtime.

The workload is one request at a time, deterministic sampling, 64 output tokens,
and a prefix-caching explanation prompt padded to 128, 1024, or 3072 tokens.
Each point has one warmup followed by three measured repeats. Acceptance means
accepted draft tokens divided by proposed draft tokens; the mean acceptance
length includes the bonus token. Decode throughput excludes the first streamed
token and prefill, while end-to-end throughput includes them. Cross-request
prefix reuse is absent: the current runner warns that draft-group identification
disables it for MTP; baseline requests use unique cache salts.

| Prompt tokens | MTP draft acceptance | Mean acceptance length | MTP decode tok/s | MTP end-to-end tok/s |
| --- | --- | --- | --- | --- |
| 128 | 71.15% | 2.423 | 7.822 | 7.609 |
| 1024 | 78.00% | 2.560 | 8.106 | 7.096 |
| 3072 | 66.67% | 2.333 | 7.396 | 5.225 |

| Prompt tokens | No-MTP decode tok/s | No-MTP end-to-end tok/s | First output mismatch (1-based) |
| --- | --- | --- | --- |
| 128 | 3.505 | 3.492 | 32 |
| 1024 | 3.496 | 3.377 | 16 |
| 3072 | 3.483 | 2.987 | 17 |

Rates are medians; MTP uses two draft tokens with adaptive verification disabled.
All repeats within each mode/context returned the same token IDs, but MTP and
non-MTP greedy outputs are not identical. The cause has not been isolated, so
these measurements do not establish output equivalence or task accuracy and
must not be treated as a correctness-qualified speedup. The gfx1201 route remains
opt-in. Sampled GPU states were 3144 MHz / 287 W with MTP and 3035 MHz / 308 W
without MTP, both at 100% utilization; clocks were not fixed.

Target branch: `feat/turboquant-gfx1201-k8v4-mtp`

Baseline at plan creation: `0836ecb414ce0d092467f049938f1c249371ba09`

Target profile:

- ROCm / AMD `gfx1201` (Radeon AI PRO R9700 class)
- TurboQuant cache preset `turboquant_k8v4`
- head dimension `D=256`
- GQA group size `Hq / Hk = 6`
- fast fused decode first, then multi-token prediction (MTP/spec-decode)

Known end-to-end baseline supplied for this work:

- short-context safe path: approximately 29.5 tok/s
- 32K-context safe path: approximately 17.7 tok/s
- final end-to-end objective: 40-60 tok/s where model compute and MTP acceptance permit it

This document is deliberately prescriptive. The implementation agent should follow the phase order and gates below rather than broadening the work into a general TurboQuant rewrite.

## 1. Current state and important invariants

`turboquant_k8v4` means FP8 keys plus 4-bit values. The K8 path does not need Hadamard query rotation, Lloyd-Max centroids, pair LUTs, or key norm correction. Those facilities belong to the MSE-key modes and must not be carried into the gfx1201 K8/V4 fast kernel unless profiling proves a shared abstraction is free.

The existing FlyDSL fast decode path is a gfx950/CDNA4 implementation for D=128 MSE4/V4. It is useful as a structural reference for partitioning and reduction, but it is not the implementation base for this target. Do not attempt to make the gfx950 FlyDSL kernel run on gfx1201.

The existing SoA Triton implementation is the closest functional reference for the desired cache-access pattern. The branch head already fixes the critical invariant that an SoA-written cache must never be read by an AoS decoder, and it also makes the FP8 SoA store promote input values to fp32 before the FP8 cast to match the established AoS semantics. Preserve both properties.

At present, `TurboQuantAttentionImpl._soa_store` follows FlyDSL availability. On gfx1201 FlyDSL is unavailable, so the current safe path normally remains AoS. A gfx1201 fast kernel may choose SoA, but the layout selection must be made once, before cache population, and remain stable for the lifetime of the layer/cache. Never dynamically switch an already-populated cache between AoS and SoA readers.

The target builder conditionally enables `supports_spec_as_decode=True` only for
the exact profile with the native multi-token decoder. Other TurboQuant profiles remain single-token builders. Enabling this capability changes the meaning of the decode batch: `num_decode_tokens` can become larger than `num_decodes`, so the token/request distinction must remain explicit in metadata and workspace sizing.

The current thin SoA decode adapter synthesizes `[0, 1, ..., B]` using `torch.arange` on every invocation. The gfx1201 fast path must not use that adapter. It should consume `attn_metadata.query_start_loc` directly; this is required for MTP anyway and also removes an avoidable per-layer launch/allocation in the one-token case.

## 2. Non-negotiable correctness rules

1. Cache layout decides the reader. If the layer selected SoA storage, every fallback reader for that layer must be SoA-aware. If the layer selected AoS, never dispatch an SoA reader.
2. The gfx1201 fast path is strictly gated to the supported profile until explicitly expanded: ROCm + gfx1201 + D=256 + GQA=6 + FP8 K + V4. Sinks and sliding-window attention remain on a safe fallback initially.
3. Keep the current safe path fully usable. Any unsupported shape, feature, block size, dtype, or failed fast-path eligibility check must fall back without changing cache interpretation.
4. Do not enable `supports_spec_as_decode=True` until a query-start-location-aware multi-token kernel passes correctness and CUDA/HIP graph replay tests.
5. Do not allocate shape-dependent scratch buffers during graph replay. Reserve or reuse workspace using stable base pointers.
6. Preserve the FP8 store conversion semantics fixed by branch head `0836ecb...`.
7. Performance changes must be benchmarked independently from correctness changes. Do not combine a large routing change and a large kernel rewrite in the same implementation step.

## 3. Implementation map

### Backend integration

Primary file: `vllm/v1/attention/backends/turboquant_attn.py`

Anchors:

- `TurboQuantMetadataBuilder.__init__`: keep `supports_spec_as_decode=False` through the single-token phases. Change it only in the MTP phase.
- `TurboQuantMetadataBuilder._reserve_workspace`: MTP will require token-capacity-aware scratch sizing, not merely `max_num_reqs`.
- `TurboQuantAttentionImpl.__init__`: establish a static target-profile/fast-path decision and static cache-layout decision here.
- `TurboQuantAttentionImpl._store_kv`: SoA selection for the target must be consistent with the decision made in `__init__`.
- `TurboQuantAttentionImpl._decode_attention`: dispatch the dedicated gfx1201 K8/V4 kernel before the generic SoA/AoS decode paths. Pass the existing `query_start_loc`; do not route through the thin one-token SoA adapter.
- `TurboQuantAttentionImpl._prefill_attention`: continuation-prefill must continue to use a reader matching the selected cache layout. Do not route continuation through the new kernel until its causal multi-token behavior is explicitly validated.
- `TurboQuantAttentionImpl._dispatch_decode_soa`: keep as the safety fallback for an SoA-selected layer. The dedicated fast path should not pay its `inspect.signature()` overhead.

`vllm/v1/attention/backends/turboquant_gfx1201_k8v4_mtp.py` pins the exact
eligibility contract. Production integration lives in `turboquant_attn.py` and
selects the SoA layout before cache population when the opt-in is enabled.

### Kernel implementation

Primary new file: `vllm/v1/attention/ops/turboquant_soa/triton_turboquant_decode_gfx1201_k8v4.py`

The scaffold defines two intended entry points:

- single-token fused decode
- query-start-location-aware multi-token/MTP decode

Reference implementations:

- `triton_turboquant_unified_attention.py`: SoA address calculation, query-start-location mapping, split-K reduction pattern, causal masking for multiple query tokens.
- `triton_turboquant_decode_v2.py`: grouped-Q stage-1 structure, `exp2` online softmax, fixed split workspace and shared stage-2 reducer.
- `triton_turboquant_store.py`: authoritative SoA store layout and FP8 representation.

Do not add MSE-key branches to the gfx1201 K8/V4 specialization.

### Tests

Extend `tests/quantization/test_turboquant.py` or add a focused GPU test file only after the kernel exists. The first tests should exercise store -> decode round trips and compare the fast path with the existing safe path/reference attention.

### Benchmark harness

Prefer a focused benchmark script under the existing benchmark/test conventions. The harness must measure kernel time separately from end-to-end generation speed and must allow forcing safe vs fast dispatch without changing cache semantics mid-run.

## 4. Target SoA layout for K8/V4 D=256

For the target profile:

- key data: 256 bytes/token/head (one FP8 byte per dimension)
- value data: 128 bytes/token/head (two 4-bit values per byte)
- value metadata: scale fp16 + zero fp16 = 4 bytes/token/head
- total logical storage: 388 bytes/token/head

In the existing SoA convention, the block data region contains K data + V packed data, or 384 bytes/token/head. The block metadata region contains two fp16 fields per token/head, `V_SCALE` and `V_ZERO`. The total storage remains 388 bytes/token/head; only the within-block arrangement changes.

The specialized decoder must derive strides from the actual cache tensor/block size, but D, GQA, K format and V format should be compile-time constants in the first implementation. This is intentional specialization, not a generic replacement.

## 5. Phase 0 - lock the baseline and profiling method

Before changing runtime behavior, record a reproducible baseline from the branch head.

Measure at least context lengths 128, 1K, 4K, 8K, 16K and 32K. Prioritize batch/request count 1 because it exposes decode latency and is the important MTP case; add the normal serving batch sizes used in practice after the B=1 path is understood.

For each point record:

- end-to-end generated tok/s
- per-token median/p95 latency after warmup
- TurboQuant store time
- attention/decode kernel time by layer or aggregated layer time
- stage-1 and stage-2 time separately for split-K kernels
- launch count per generated token
- GPU clock/power state during the run

Use repeated runs and report the median. Do not compare a cold compile/capture run with a warmed run.

The implementation agent must preserve the supplied baseline labels (about 29.5 tok/s short and 17.7 tok/s at 32K) but should replace them with exact locally reproduced numbers before claiming a speedup.

Exit gate: a checked-in or recorded benchmark command/config can reproduce the safe path without changing model outputs.

## 6. Phase 1 - remove launcher overhead without changing math

This phase is deliberately small.

For the future gfx1201 fast route:

- pass `attn_metadata.query_start_loc` directly to the kernel launcher
- do not construct `torch.arange(B + 1)` per layer/step
- do not use `_dispatch_decode_soa()` reflection on the fast path
- do not build centroid/pair-LUT work for K8
- reuse caller-provided output/scratch buffers

Do not change the current default route yet if the specialized kernel is not present.

Exit gate: output is unchanged versus the safe path and there is no measurable regression. Any gain is useful but no minimum speedup is required in this phase.

## 7. Phase 2 - implement the specialized single-token fused decode

Implement the single-token entry point in `triton_turboquant_decode_gfx1201_k8v4.py` first.

Initial kernel shape:

- grid conceptually over `(request, kv_head, split)`
- one program handles all six Q heads sharing a KV head
- initial `BLOCK_M=16` with only six valid GQA rows; this is conservative and matches shapes known to compile through `tl.dot`
- initial `TILE_SIZE=16`
- start with `num_stages=1` on ROCm
- stage 1 writes online-softmax partial output + LSE/max state to reusable scratch
- stage 2 reuses an existing numerically compatible reducer where possible

K path:

- load FP8 key bytes directly from the SoA data region
- reinterpret using the same FP8 flavor selected by `_use_fp8_e4b15`
- convert in registers to the dot input dtype initially
- compute grouped QK with `tl.dot`
- do not rotate Q and do not access Pi/PiT/centroids

V path:

- load each packed byte once
- extract low/high nibbles and interleave to D=256
- load `V_SCALE` and `V_ZERO` from the SoA metadata region using aligned u16 loads
- reconstruct V in registers
- accumulate P*V using `tl.dot`

Softmax:

- use an online softmax
- prefer the established `exp2` formulation from the v2 kernel to avoid unnecessary transcendental overhead
- maintain fp32 accumulators for max/sum/output

Do not implement MTP in this phase. The single-token launcher should nevertheless accept `query_start_loc` in its public contract so the later multi-token path does not require an incompatible API redesign.

Exit gates:

- numerically matches the safe path/reference within the tolerance established by existing TurboQuant GPU tests
- no NaNs/infs across the context sweep
- correct on non-contiguous paged block tables and final partial blocks
- graph capture/replay does not allocate or change pointers
- fast kernel shows a useful kernel-time improvement before it is wired as default

## 8. Phase 3 - wire a static gfx1201/SoA fast route

Only after the Phase 2 kernel passes correctness, add the backend route.

Eligibility must initially require all of:

- ROCm
- exact arch `gfx1201`
- `head_size == 256`
- `num_kv_groups == 6`
- `tq_config.key_fp8 is True`
- `tq_config.effective_value_quant_bits == 4`
- no attention sinks
- no sliding window
- supported block size proven by tests (start with 16/32 if that is what was tested)

The fast-path enable decision must be layer/session static. If enabled, select SoA storage before the first KV write. If a runtime condition prevents use of the specialized kernel after SoA was selected, fall back to the existing SoA Triton reader, never the AoS reader.

Keep an explicit opt-in while tuning. The opt-in must be read once during initialization, not inside the per-token hot path. After performance/correctness gates are met, it can become automatic for the exact target profile.

Exit gate: safe and fast routes can be A/B tested from process start, produce matching results, and never mix cache layouts.

## 9. Phase 4 - gfx1201 tuning

Do not inherit MI300X tuning constants. The R9700/gfx1201 path has 32 CUs and wave32 behavior; its occupancy/parallelism balance is different from gfx94x/gfx950.

Sweep only after the specialized kernel is correct.

Required sweep dimensions:

- `TILE_SIZE`: 16, 32, 64 where compilation/resource usage permits
- split count: 1, 2, 4, 8, 16, and optionally 32 for the longest context
- `num_warps`: 2 and 4 first; test 8 only if resource use permits
- `num_stages`: 1 and 2
- dot input type: fp16 vs bf16 if both are numerically acceptable
- `BLOCK_M`: keep 16 as the baseline; test a smaller padded grouped-Q shape only if Triton generates valid/effective dot code

Tune short and long contexts separately. The split-K threshold must be determined on gfx1201; do not retain the generic/MI300-oriented 1024-token threshold without measurement.

For each winning candidate capture compiler/resource information when available: VGPR usage, LDS usage, waves/occupancy, and whether time is dominated by memory, VALU unpack/dequant, MFMA/dot, or stage-2 reduction.

Decision gate for staying with Triton: aim for at least a 1.25x median attention-kernel speedup over the safe path across 4K-32K without more than a 5% regression in the short-context points. If Triton cannot reach this after the bounded sweep, stop tuning Triton and implement a gfx1201-specific HIP/AOT kernel using the same public launcher contract. Do not spend unbounded time on heuristic changes.

## 10. Phase 5 - implement true multi-token/MTP decode

After the single-token route is stable, extend the kernel to support multiple query tokens per request.

The key semantic change is that the decode input is no longer `B requests == B query tokens`. The kernel must use `query_start_loc` to map query tokens to requests. For request `r`:

- `q_len = query_start_loc[r+1] - query_start_loc[r]`
- `context_len = seq_len[r] - q_len`
- query row `q_pos` may attend through absolute KV position `context_len + q_pos`

Share each loaded/dequantized K/V tile across as many query rows of the same request/GQA group as possible. This reuse is the main reason to implement MTP inside the fused kernel rather than expanding each speculative token into an independent single-token launch.

Support the maximum decode query length implied by vLLM's speculative configuration, including the parallel-drafting multiplier used by `_init_reorder_batch_threshold`.

### Metadata-builder switch

Only after the multi-token path passes tests, change `TurboQuantMetadataBuilder.__init__` to initialize the reorder threshold with `supports_spec_as_decode=True` for configurations where the new kernel is usable. Do not enable this globally for unsupported TurboQuant profiles unless they also have a correct multi-token decoder.

A profile-conditional builder decision is preferable to silently sending MTP-shaped batches to a one-token backend.

### Workspace sizing

Current decode scratch reservation is request-sized. MTP partial outputs are token-sized. When `max_query_len > 1`, reserve enough rows for the maximum number of decode query tokens that a captured/replayed batch can contain, not merely `max_num_seqs`.

Derive a bounded maximum from the scheduler/capture configuration and the effective speculative decode threshold. Keep base pointers stable for graph replay. Do not solve this by allocating from `num_actual_tokens` during the hot path.

### Forward-path audit

Audit every place that currently assumes `query.shape[0] == num_decodes`. In mixed batches, `num_decode_tokens` is the token split point and `num_decodes` is the request count. Preserve that distinction through the new launcher and workspace indexing.

Exit gates:

- q_len=1 remains identical to the tuned single-token result
- q_len>1 causal outputs match reference attention/safe dequantized attention
- mixed decode+prefill batches split correctly
- graph capture/replay works for the configured speculative length
- no request/token indexing uses the wrong dimension

## 11. Phase 6 - correctness matrix

The fast route must be tested against the safe/reference path over at least:

- query dtype fp16 and bf16 if both are supported by the model path
- block sizes 16 and 32 initially
- sequence lengths around block/tile boundaries: 1, 15, 16, 17, 31, 32, 33 and representative long lengths
- non-contiguous physical block tables
- multiple KV heads with GQA exactly 6
- K FP8 representation used by the actual gfx1201 runtime
- one-token decode
- MTP query lengths from 2 through the configured maximum representative values
- prefix-cache / continuation scenarios
- mixed decode+prefill scheduling
- CUDA/HIP graph capture and replay

Compare both final attention output and end-to-end generated tokens for deterministic prompts. A speed result is invalid if the generated stream diverges beyond the expected TurboQuant approximation from the safe path.

## 12. Phase 7 - end-to-end performance acceptance

Report three separate numbers so gains are attributable:

1. attention kernel latency/speedup versus safe TurboQuant
2. non-MTP end-to-end tok/s versus the approximately 29.5 short / 17.7 32K baseline
3. MTP end-to-end accepted tok/s, together with acceptance rate and average accepted tokens per verification step

The project objective is 40-60 tok/s end-to-end where achievable. Do not claim the objective from kernel microbenchmarks alone. If the specialized attention kernel is fast but model GEMMs become the dominant limit, state that explicitly and stop modifying attention code without evidence.

## 13. Fallback and rollback policy

The current safe implementation is the oracle and permanent fallback during development.

If the fast-path eligibility check fails before cache creation, use the current cache layout/path unchanged.

If the layer selected SoA fast mode and later encounters an unsupported decode feature, use the existing SoA Triton decoder. Never fall back to AoS for an SoA-populated cache.

If MTP causes instability, disable spec-as-decode while retaining the proven single-token gfx1201 kernel. MTP and single-token optimization must be independently reversible.

## 14. Commit discipline for the implementation agent

Keep the work bisectable. Recommended implementation commits:

1. benchmark/profiling harness and exact reproduced baseline
2. specialized K8/V4 D256 GQA6 single-token kernel, not routed by default
3. strict gfx1201 eligibility + static SoA routing behind opt-in
4. gfx1201 tuning constants selected from recorded sweep
5. multi-token query-start-location kernel and token-sized workspace
6. conditional `supports_spec_as_decode=True` + MTP integration
7. remove/flip experimental opt-in only after all gates pass

Each performance commit should contain the before/after benchmark evidence in its commit message or an adjacent result note.

## 15. Guardrails for Codex/Luna Max

Do not broaden scope into generic TurboQuant cleanup, new quantization presets, other GPU architectures, MSE-key optimization, FlyDSL porting, sinks, sliding-window support, or unrelated vLLM refactors until the target profile reaches the performance/correctness gates.

Do not change public behavior merely to make a microbenchmark easier.

Do not enable speculative decode by changing one boolean and then repair failures reactively. Implement query-start-location-aware decode and workspace sizing first.

Do not create a second incompatible cache layout. Reuse the existing SoA layout exactly.

Do not keep tuning a Triton kernel indefinitely. Use the Phase 4 decision gate to decide whether a HIP/AOT implementation is justified.

When uncertain, prefer the smallest change that preserves the safe fallback and produces a measurable result on the R9700.

## 16. Definition of done

This work is complete when all of the following are true:

- the exact gfx1201 / D256 / GQA6 / K8V4 profile automatically or explicitly selects a proven fast route
- unsupported profiles/features retain the existing safe route
- no AoS/SoA cache mismatch is possible
- single-token decode is materially faster on the R9700 across long contexts
- MTP/spec-decode batches are handled natively using `query_start_loc`
- HIP graph replay uses stable workspaces and performs no hot-path allocations
- correctness tests cover block boundaries, paged block tables, continuation/prefix use, and MTP
- benchmark results clearly separate kernel speedup, non-MTP end-to-end speedup, and MTP acceptance-driven speedup
- the final end-to-end result is compared honestly against the 40-60 tok/s project objective
