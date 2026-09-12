# gfx1201 long-prefill selective-port plan

Status: implementation plan and inert scaffolding only

Base revision: `787189270cc97f0671e2f2f4aa33a506b07df335`

Target branch: `feat/gfx1201-radiance-prefill`

External reference pin: `magiccodingman/vllm-radiance` at `adf9e1f1c9529dd6c971b223a961833376dbd524`.

This plan begins after the W4A8 decode prototype in the parent branch failed its A3 performance gate. It does not reopen that decode decision. Long-prompt prefill is evaluated independently because the previous Radiance setup showed much higher long-context prefill throughput under a different kernel/cache stack.

## Historical local reference

Operator-recorded TP1 Radiance result for Quark AWQ MXFP4 Qwen3.8-27B:

- TP1, max sequence 1, about 131K context;
- MTP8, R4D attention, FP8 KV;
- short decode about 47.1 tok/s;
- prefill about 2,100 tok/s at 32K, 1,735 tok/s at 64K, 1,326 tok/s at 120K;
- early/middle/late needle probes passed at 32K/64K/120K;
- short accepted/drafted about 49%, long logs about 81--100%;
- about 32.3 GB VRAM, peak about 267 W, hotspot 91 C, VRAM 81 C;
- no OOM/device loss.

These numbers are a north-star, not an adoption threshold. The historical lane used FP8 KV and MTP8; the validated local lane uses K8/V4 and MTP2. Final qualification must report the ratio to the historical 32K/64K/120K numbers, but matched local baselines decide adoption.

## What failed A3 means

`787189270c` validated a row-major Triton W4A8 decode prototype numerically but it was slower than the current software-fused MXFP4 decode kernel on every measured production shape (six-shape geometric mean 0.755x). A4/A5 in the previous plan correctly stopped.

This does not disprove either long-prefill hypothesis:

1. a native gfx1201 FP8-WMMA large-M W4A8 kernel may still beat the large-M emulation path;
2. long continuation attention can avoid dequantizing/materializing the complete cached K/V prefix on every chunk.

Do not enable or retune the failed small-M W4A8 route in this plan.

## Current prefill path

The base TurboQuant path behaves as follows:

1. first-chunk prefill uses raw Q/K/V and prefers flash-attention, otherwise SDPA;
2. continuation `q_len <= 128` consumes the K8/V4 cache directly through the TurboQuant reader;
3. larger continuation chunks dequantize the cached prefix into FP16 workspace, rebuild full `k_full`/`v_full`, then run flash-attention or SDPA.

The validated long-context run already exposed multi-GiB workspace pressure at 32K for large continuation chunks. Therefore continuation attention is the first prefill hypothesis; decode percentages must not be reused to prioritize prefill.

## Global rules for Luna

- One hypothesis per commit; never combine continuation attention, W4A8 prefill, raw attention, GDN, MTP-depth, compiler, or cache-format changes.
- Keep K8/V4, the current software-fused MXFP4 backend, hybrid prefix reuse, and compilation-disabled `FULL_DECODE_ONLY` as rollback baseline.
- Official qualification remains MTP2. MTP8 is historical comparison only.
- Preserve the K8/V4 cache contract. FP8 KV is diagnostic only.
- All new production paths start behind explicit default-off opt-ins and fail closed to existing code.
- No full dense attention-score matrix.
- No new full-prefix K/V materialization in the direct-continuation phase.
- Record wave size, VGPR, SGPR, LDS/shared, scratch/private segment, launch geometry for adopted kernels.
- Kernel A/B uses preallocated outputs, warmed compilation, rotating order, raw samples, identical source tensors.
- Same-process/same-load A/B is required for model-level claims below 5%.
- Arithmetic-changing paths require model-quality gates; token hashes alone are insufficient.
- Stop when a phase gate fails; do not turn on unrelated Radiance features to compensate.

## P0: provenance gate

Before copying external implementation code, record exact source commit/file, authorship/provenance, and license. If a compatible license cannot be established, implement from published algorithm/behavior and local measurements rather than transliterating source.

Relevant external references are `radiance_mxfp4.py`, `radiance_mxfp4_fp8.hip`, `radiance_r4d_attn.py`, `radiance_gdn.py`, `docs/MXFP4_W4A8_R9700.md`, and `docs/MXFP4_RX5_FP8KV_CONTINUATION.md` at the pinned commit above.

## P1: freeze and profile the long-prefill baseline

Run the parent implementation with all new prefill opt-ins absent.

Official lane:

- Quark MXFP4 target, TP1;
- TurboQuant K8/V4;
- current software-fused gfx1201 MXFP4 decode path;
- MTP2, adaptive verification disabled;
- compilation disabled + `FULL_DECODE_ONLY`;
- max sequence count 1;
- cold/non-reused prompts for prefill throughput.

Measure 4K/8K/16K/32K and, when baseline completes, 64K/120K. Sweep scheduler chunk sizes 128/256/512/1024 where memory permits. Freeze the exact prefill-throughput formula before comparing candidates.

For every point record prompt tokens, prefill tok/s, TTFT, fixed-64-output E2E, peak allocated/reserved/physical VRAM, completion/failure reason, selected attention path, speculative counters, and clock/power/temperature when available.

Collect correlation-based prefill device time split into at least:

- MXFP4 weight dequant/dense GEMM;
- TurboQuant store;
- cached-prefix dequantization;
- full K/V copies/conversions;
- attention compute;
- GDN/FLA prefill;
- norm/activation/indexing;
- other.

P1 ends when a machine-readable summary identifies the largest two 32K prefill cost centers. 64K/120K may fail; preserve first failure stack and peak memory.

## P2: direct K8/V4 continuation prefill

Integration points:

- `TurboQuantAttentionImpl._prefill_attention`;
- `TurboQuantAttentionImpl._continuation_prefill`;
- scaffold `vllm/v1/attention/ops/turboquant_soa/gfx1201_prefill.py`;
- benchmark `benchmarks/kernels/benchmark_turboquant_gfx1201_prefill.py`;
- GPU tests `tests/kernels/attention/test_turboquant_gfx1201_prefill.py`.

### P2.1 reuse-before-rewrite

Before a new kernel, benchmark the existing SoA-aware direct TurboQuant reader beyond the current 128-token threshold using captured production tensors and synthetic controls:

- q_len 64/128/256/512/1024;
- cached lengths 4K/8K/16K/32K/64K where feasible;
- D=256, GQA=6;
- production block/layout.

Compare current full-dequant continuation, the generic SoA direct reader with a benchmark-only widened threshold, and the specialized gfx1201 K8/V4 multi-token reader if its metadata/workspace contract permits those widths. Do not change production threshold in this commit.

Prefer reuse if q_len=512 / 32K is at least 1.25x faster than current continuation and needs no more than 384 MiB additional workspace.

### P2.2 dedicated streaming kernel

Implement only if reuse cannot pass cleanly.

Initial scope:

- ROCm gfx1201;
- K8/V4 SoA cache;
- D=256, GQA6;
- causal BF16 query/current-chunk K/V;
- no sinks/window;
- one request per launch first.

Tensor contract:

- query `[Q,Hq,256]` BF16;
- raw current `key_chunk`/`value_chunk` `[Q,Hk,256]` BF16;
- existing SoA K8/V4 paged cache + one block table;
- `seq_len = cached_len + Q`;
- output `[Q,Hq,256]`.

To preserve current large-continuation semantics, prefix positions `[0,cached_len)` come from compressed cache while current positions `[cached_len,seq_len)` come from raw K/V. Query row `q` sees through `cached_len + q`. Do not silently read current K/V back from the quantized cache in the first adopted implementation.

Kernel strategy:

- tile multiple query positions and six GQA heads so each dequantized KV tile is reused;
- dequantize K8/V4 only in register/LDS tiles immediately consumed by score/value math;
- online softmax; no full score matrix;
- no full-prefix K/V output buffer;
- start without split-KV; add it only if long-context occupancy requires it;
- if split-KV is needed, cap one-sequence 131K persistent scratch at 256 MiB unless separately reviewed.

Numerical gate:

- synthetic small cases compare candidate and current path to FP64 oracle; candidate max-abs/RMSE may not exceed current error by more than 10%;
- captured production cases record per-layer max-abs/RMSE and final logits;
- no unexpected non-finite values;
- causal first/last-row and cache-block-boundary tests are mandatory.

Performance/adoption gate:

- q_len=512 / 32K continuation device time at least 1.5x faster than current full-dequant continuation;
- cold 32K prefill at least 20% faster in same-load A/B;
- peak workspace lower than current dequant + `k_full/v_full` path and no O(cached_len*Hk*D) full K/V buffer;
- 64K/120K no regression if baseline completed; if baseline failed only from workspace and candidate completes, record as functional win and continue qualification.

Suggested opt-in after benchmark pass: `VLLM_TQ_GFX1201_PREFILL`, default False.

## P3: native FP8-WMMA W4A8 large-M only

P3 is independent from failed A3. Small-M calls remain on the current software-fused MXFP4 backend.

The existing `launch_gfx1201_w4a8_prefill()` placeholder is the Python boundary. The functional candidate must use a HIP/native route proven by disassembly to emit the intended gfx1201 FP8 matrix instruction. A Triton route that expands FP8/MXFP4 to BF16 and runs BF16 WMMA is not the target.

Suggested HIP source: `csrc/quantization/gfx1201/mxfp4_w4a8_prefill.hip`, registered through `csrc/ops.h` / `csrc/torch_bindings.cpp` and appended to the ROCm `_C` target unless a separate ABI design says otherwise.

Initial contract:

- gfx1201 only;
- Quark/OCP MXFP4 weight, group32;
- BF16 source activation -> dynamic per-row FP8 E4M3;
- BF16 output;
- bias unsupported;
- large-M only; threshold from benchmark.

Do not assume the A3 reference scale rule is final. Reconcile with the chosen runtime FP8 quantization semantics and add zero/tiny/saturation/scale-boundary/non-finite tests before timing.

Benchmark M=128/256/512/1024/2048/3072/4096 on all dense prefill shapes, reporting quantization, GEMM, and combined time.

Adoption gate:

- production-weighted `quant + GEMM` at least 1.25x faster than current large-M emulation;
- cold 32K prefill at least 10% faster with P3 alone;
- no small-M decode regression because dispatch is unchanged;
- 308-case quality gate: HumanEval remains 164/164 and neither GSM8K nor MMLU aggregate may fall below validated baseline without explicit review;
- record 12-prompt logits, MTP acceptance, and target inputs; bitwise W4A4 equality is not required.

Suggested separate opt-in: `VLLM_ROCM_USE_GFX1201_MXFP4_W4A8_PREFILL`, default False.

## P4: raw first-chunk prefill attention

Re-profile after P2/P3. Implement only if raw/first-chunk attention is at least 10% of remaining 32K prefill kernel time or if the environment lacks a working efficient flash-attention path and SDPA is dominant.

Initial narrow profile: D=256, GQA6, causal BF16 Q/K/V, one request, no sinks/window/alibi/soft-cap. It consumes raw Q/K/V and does not change TurboQuant cache layout.

Gate: attention kernel at least 1.3x faster on production shapes, cold 32K prefill at least 5% faster alone, no worse error envelope versus FP64 oracle, and 308-case quality gate before composition.

Suggested opt-in: `VLLM_TQ_GFX1201_RAW_PREFILL`, default False.

## P5: GDN prefill fusion

Re-profile after P2--P4. Proceed only if GDN/FLA prefill is at least 10% of remaining 32K prefill kernel time.

Do not alter hybrid prefix-cache ownership/hash/replay semantics established by `ef9433f115`.

Confirm live GDN geometry before writing the kernel. Radiance uses head-K128/head-V128/chunk64 and fuses the high-traffic WY/state-scan/output portion; treat that as a hypothesis, not an assumed local contract.

Scaffold: `vllm/model_executor/layers/mamba/ops/gfx1201_gdn_prefill.py`.

Correctness compares both sequence output and final recurrent state for zero/captured initial state, multiple chunks, partial final chunk, and prefix-cache replay.

Gate: fused subsection at least 1.3x faster than replaced kernels, cold 32K prefill at least 5% faster alone, tightly bounded recurrent-state error without long-sequence drift, 308-case quality gate, and 3000-token/2976-hit prefix-reuse probe.

Suggested opt-in: `VLLM_ROCM_USE_GFX1201_GDN_PREFILL`, default False.

## P6: compose and qualify only successful phases

Do not compose rejected phases. Enable accepted components one by one and record incremental/cumulative deltas.

Final matrix:

- cold 32K/64K/120K prompts, max model length around 131K where memory permits, max sequence count1;
- official MTP2 lane;
- fixed 64 outputs with EOS ignored;
- early/middle/late needle retrieval at each context;
- 3000-token repeated-prefix probe must retain 2976-token hit;
- 308-case GSM8K/HumanEval/MMLU gate;
- 24-minute serving soak after stable composition.

Report prefill tok/s, TTFT, decode/E2E rates, acceptance by draft position and mean accepted length, semantic/token variation, peak allocated/reserved/physical VRAM, power/temperature, OOM/abort/preemption/device-loss counts, and post-composition kernel-time breakdown.

After official qualification, optionally run MTP8 for historical comparison. Print ratios to historical 2100/1735/1326 tok/s at 32K/64K/120K, but matched baseline gates decide adoption.

## Diagnostic FP8-KV control

If qualified K8/V4 remains far below historical Radiance, run an isolated FP8-KV/Radiance control to identify whether the gap is cache-format/attention, W4A8/GDN, runtime/chunking, or compiler related. Keep that diagnostic out of production commits unless a later design explicitly changes the cache contract.

## Planned commit sequence

1. `[gfx1201] Plan long-prefill Radiance follow-up` — this plan + inert P2/P5 scaffolds; runtime unchanged.
2. `[gfx1201] Profile K8V4 long prefill` — P1 helpers/report only.
3. `[TurboQuant] Audit wide-query direct prefill reuse` — P2.1 benchmark/decision only.
4. `[TurboQuant] Stream gfx1201 K8V4 continuation prefill` — P2.2 only if needed and gated.
5. `[MXFP4] Add gfx1201 native W4A8 prefill` — P3, independent of failed A3.
6. `[TurboQuant] Add gfx1201 raw prefill attention` — P4 only if profile-gated.
7. `[ROCm] Fuse gfx1201 GDN prefill` — P5 only if profile-gated.
8. `[gfx1201] Qualify long-prefill composition` — P6 report/quality/needles/prefix/soak; defaults unchanged.

The scaffold modules are intentionally not imported. Until their own functional phases, candidate checks return False and launch functions raise `NotImplementedError`.
