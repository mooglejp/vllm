# gfx1201 long-prefill selective-port plan

Status: P0 provenance gate recorded; P1 baseline profile complete; P2.1 reuse evaluation complete; P2.2 current candidates evaluated and rejected; P3 current native W4A8 candidate speed gate failed; default dispatch unchanged

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

P0 provenance decision (2026-09-12): The pinned `magiccodingman/vllm-radiance@adf9e1f1c9529dd6c971b223a961833376dbd524` was checked before any implementation copy. Its GitHub metadata and repository root do not expose a compatible license; the README credits StillDeadcode/libr4d and ggz14/Radiance MXFP4 but does not grant a source-code license. No external implementation was copied into this branch. The P1 helper and profiling instrumentation are clean-room work based on the public behavior in this design and local traces. Any later port remains blocked until a compatible license and exact file-level provenance are recorded.

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

### P1 execution record (2026-09-12)

The run used revision `4d33849ec927fb28e13bf5d1c13dd23b6efe0f2f` on an AMD Radeon AI PRO R9700 (gfx1201), ROCm 7.14, TP1, and the official Quark MXFP4 model. The lane was MTP2 with adaptive verification disabled, TurboQuant K8/V4, compilation disabled with `FULL_DECODE_ONLY`, max model length 131,072, max sequence count 1, random `cache_salt` per request, and fixed 64 output tokens with EOS ignored. No new prefill opt-in was set.

The throughput formula was frozen as `prefill_tok_s = prompt_tokens / TTFT_s`; fixed-output E2E is `completion_tokens / elapsed_s`. The chunk-128 baseline completed through 32K:

| prompt | prefill tok/s | TTFT (s) | fixed-64 E2E tok/s | result |
| ---: | ---: | ---: | ---: | :--- |
| 4,096 | 310.43 | 13.1945 | 3.2449 | complete |
| 8,192 | 259.89 | 31.5205 | 1.6937 | complete |
| 16,384 | 192.25 | 85.2236 | 0.6973 | complete |
| 32,768 | 127.41 | 257.1877 | 0.2424 | complete |

The scheduler sweep was run under the same lane. Raw JSONL preserves prompt hashes, output hashes, speculative counters, TTFT, E2E, and decode rates at `/tmp/tq-long-prefill-p1.zMusXN/{chunk128,chunk256,chunk512-partial,chunk1024-partial}.jsonl` (the corresponding cache copies are under `/cache`). Completed points were:

| chunk | prompt | prefill tok/s | TTFT (s) | fixed-64 E2E tok/s | result |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 128 | 4,096 | 310.43 | 13.1945 | 3.2449 | complete |
| 128 | 8,192 | 259.89 | 31.5205 | 1.6937 | complete |
| 128 | 16,384 | 192.25 | 85.2236 | 0.6973 | complete |
| 128 | 32,768 | 127.41 | 257.1877 | 0.2424 | complete |
| 256 | 4,096 | 550.97 | 7.4341 | 4.7621 | complete |
| 256 | 8,192 | 471.83 | 17.3621 | 2.7083 | complete |
| 256 | 16,384 | 372.69 | 43.9620 | 1.2797 | complete |
| 256 | 32,768 | 247.86 | 132.2025 | 0.4610 | complete |
| 512 | 4,096 | 739.36 | 5.5400 | 5.3148 | complete |
| 512 | 8,192 | 590.06 | 13.8833 | 3.1355 | complete |
| 512 | 16,384 | 441.97 | 37.0708 | 1.4670 | complete |
| 1024 | 4,096 | 785.83 | 5.2123 | 5.5749 | complete |
| 1024 | 8,192 | 616.45 | 13.2890 | 3.2681 | complete |

The first sweep failures were retained. At chunk 512 / 32K, `torch.OutOfMemoryError` came from `TurboQuantAttentionImpl._continuation_prefill` at `F.scaled_dot_product_attention`: a 290 MiB allocation was requested with 152 MiB free (30.24 GiB allocated, 821.25 MiB reserved but unallocated). At chunk 1024 / 16K, the same path requested 1.44 GiB with 1.38 GiB free (29.01 GiB allocated, 806.82 MiB reserved but unallocated). A chunk-128 64K probe exceeded the client 600 s timeout without a completion; the first failure was the client `TimeoutError`, so 120K was not attempted. These points are not treated as candidate wins.

The observed attention path was TurboQuant large-continuation full-prefix K/V dequantization, full `k_full`/`v_full` materialization, and the SDPA fallback (`F.scaled_dot_product_attention`) on this gfx1201 environment. Logs selected the existing EmulationMxfp4LinearKernel, Triton/FLA GDN prefill, and TURBOQUANT backend. A sampled active run reported physical VRAM 33,147,588,608 / 34,208,743,424 bytes at chunk 256; another active 64K probe reported 29,514,899,456 bytes, GPU busy 100%, 281 W, junction 90 C, and memory 74 C. These are samples, not claimed maxima. After each server cleanup, physical VRAM returned to 59,912,192 bytes.

The correlation profile was collected from a finite 256-iteration torch profiler run for a cold 32K request. The compressed trace is `/tmp/tq-long-prefill-p1.zMusXN/trace-full.json.gz` (192,176,586 bytes; 3,250,189,952 bytes uncompressed), with 12,050,896 trace events and 1,165,514 kernel events; all kernel correlations resolved. The machine-readable result is `/tmp/tq-long-prefill-p1.zMusXN/summary-v3.json`, generated by `benchmarks/benchmark_gfx1201_long_prefill.py`. It reports 1,084,160 prefill kernels and the following mutually exclusive split (GPU time sums, not wall time):

| prefill family | GPU ms | share | calls |
| :--- | ---: | ---: | ---: |
| attention compute | 154,590.044 | 68.793% | 4,080 |
| MXFP4 dequant / dense GEMM | 64,333.066 | 28.629% | 233,504 |
| norm / activation / indexing | 1,873.484 | 0.834% | 304,224 |
| other | 1,523.980 | 0.678% | 148,208 |
| GDN / FLA prefill | 1,186.279 | 0.528% | 98,304 |
| full K/V copy / conversion | 1,146.090 | 0.510% | 291,744 |
| TurboQuant store | 63.422 | 0.028% | 4,096 |
| cached-prefix dequantization | 0.000 | 0.000% | 0 |

The default-shape profiler attempt failed after `profiler_stop` with an EngineCore exit and a ROCTracer duplicate-flow warning; the low-overhead finite run completed and the full finite trace above was preserved despite the profiled request later exiting. This limitation is recorded rather than silently discarded.

P1 adoption gate: **pass**. The machine-readable summary identifies the largest two 32K prefill cost centers as attention compute and MXFP4 dequant / dense GEMM. No P2 or P3 production kernel was implemented or enabled; the inert scaffolds remain unchanged.

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

### P2.1 execution record (2026-09-13)

The benchmark-only harness is `benchmarks/kernels/benchmark_turboquant_gfx1201_prefill.py`. It was run against the P1 baseline source (`b8547a3572`) in the existing gfx1201 GPU container (PyTorch `2.12.0+git6bbd260`, HIP `7.2.53211`). Production threshold and dispatch code were not changed.

The controls use deterministic BF16 tensors with Hq=24, Hk=4, D=256, K8/V4 SoA storage, and block size 16. No captured model tensors were available, so these are synthetic cache-contract controls. The feasible matrix covers 4K/q64,128,256; 8K/q1024; 16K/q512; 32K/q256,512; and 64K/q64. The machine-readable output is `/tmp/p2.1-turboquant-prefill-reuse.jsonl`.

The three measured paths are deliberately separated:

- `current_large_continuation`: dequantize only the prefix into the existing FP16 workspace, append raw current K/V, and run the SDPA causal fallback.
- `generic_soa_direct_rowwise`: reuse the existing SoA reader shape with one synthetic request per query row; both prefix and current K/V are read from the quantized cache.
- `specialized_multi_token`: use the existing gfx1201 packed multi-token launcher with one request and `query_start_loc=[0,q_len]`; both prefix and current K/V are read from the quantized cache. Split counts 4/8/16 were measured.
  For q_len > 2 its existing launcher uses query blocks of 4 (q512 therefore launches 128 query blocks), so this is not one shared KV load for all 512 query tokens.

The all-quantized reference dequantizes every visible position to FP16 and then casts to the query dtype before SDPA. The raw-current reference dequantizes only the prefix to FP16 and appends raw current K/V. This records the FP16-workspace rounding separately from the current-chunk quantization difference.

Median device times (microseconds) were:

| cached / q | current large | generic best | specialized split 4 / 8 / 16 | specialized speedup vs current |
| ---: | ---: | ---: | ---: | ---: |
| 4K / 64 | 3,568 | 5,300 | 712 / 780 / 994 | 3.59--5.01x |
| 4K / 128 | 6,638 | 10,877 | 1,571 / 1,515 / 1,579 | 4.20--4.38x |
| 4K / 256 | 9,539 | 20,588 | 2,662 / 2,673 / 2,855 | 3.34--3.58x |
| 8K / 1024 | 80,996 | 159,691 | 20,503 / 21,078 / 21,408 | 3.78--3.95x |
| 16K / 512 | 69,753 | 154,550 | 18,603 / 18,748 / 19,256 | 3.62--3.75x |
| 32K / 256 | 79,520 | 151,899 | 17,340 / 17,152 / 16,962 | 4.59--4.69x |
| 32K / 512 | 145,591 | 306,269 | 34,212 / 34,368 / 34,407 | 4.23--4.26x |
| 64K / 64 | 84,918 | 76,351 | 9,341 / 9,679 / 8,798 | 8.77--9.65x |

The generic reader was slower than current large continuation except for the 64K/q64 control, where it was only 1.11x faster. The specialized reader was consistently faster, but that result is not a production adoption result because it quantizes the current chunk.

The direct-reader numerical envelope against the all-quantized reference was finite with relative L2 error 0.00266--0.00284 and maximum absolute error at most 0.00098. Against the raw-current reference, the specialized and generic paths differed by relative L2 0.00383--0.02691 and maximum absolute error up to 0.00452; this is the expected quantized-current-chunk plus FP16-workspace contract difference, not automatically a reader bug. The current large path matched its raw-current reference exactly.

Workspace accounting confirms the split warning. At 32K/q512, current-large explicit additional workspace was 274.25 MiB; specialized `mid_o` was 48.2/96.4/192.8 MiB for split 4/8/16, with total specialized additional workspace 54.2/105.0/198.5 MiB. At 8K/q1024, `mid_o` was 96.4/192.8/385.5 MiB, so split 16 alone exceeds the 384 MiB limit. Generic split scratch follows the same `[q_len,Hq,splits,D]` growth and did not produce a speed win.

P2.1 reuse adoption gate: **not adopted**. The generic reader fails the speed gate. The specialized reader passes the device-time and workspace checks on the completed common points (including 32K/q256) and is much faster on the synthetic 32K/q512 control, but it violates the current raw-current-chunk numerical contract. The P1 model run remains the authority for capacity: chunk 512/32K old path OOMed, so the synthetic 32K/q512 timing is not used as an old-path speed ratio. Likewise, the P1 64K model result remains a 600-second timeout, not an OOM; the synthetic 64K/q64 completion does not infer a 120K result.

No production threshold or default dispatch was changed. P2.2 had a default-off candidate hook for the narrow target profile; real-model speed and capacity observations are recorded, but the adoption gate was not passed. The current candidate evaluation is closed and the hook remains disabled.

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

Suggested opt-in after benchmark pass: `VLLM_TQ_GFX1201_K8V4_PREFILL`, default False.

### P2.2 implementation record

P2.2 implementation is present as a benchmarked, default-off candidate. It
reuses the gfx1201 K8/V4 stage-1 tile shape and online softmax, but selects
the input source per KV position: `[0,cached_len)` is decoded from the SoA
cache and `[cached_len,seq_len)` is loaded from raw current-chunk K/V. Both
ranges update one softmax state, and boundary loads are masked before either
cache or raw pointer is dereferenced. The first launcher is splitless and
one-request only; it has no `mid_o` scratch. The optional production hook is
limited to `VLLM_TQ_GFX1201_K8V4_PREFILL=True`, the existing target profile,
`cached_len > 0`, and `q_len > 128`. The default remains unchanged.

The benchmark now separates reference generation from candidate execution.
Math SDPA is fixed for numerical references, while the old whole-path record
uses runtime-auto SDPA and records that selection separately. A reference OOM
is retained as `reference_oom` and does not prevent candidate execution.
Split scratch is allocated one candidate at a time; calculated workspace and
allocator peak deltas are recorded independently. Each candidate's GPU output
and scratch are owned by a short-lived measurement function; only CPU timing
and error statistics escape it. This also initializes both old-path dequant
buffers before allocation, so an OOM while creating the second buffer cannot
mask the original failure. Generic whole-path timing, fixed-metadata timing,
specialized-reader timing, and the raw-current streaming candidate are
distinguished, with both the continuation pair and reader candidates ordered
alternately.

The candidate predicate now requires raw K/V shapes `[Q,Hk,256]` and unit
stride on the query, raw K/V, and launcher output's last dimension. Unsupported
shape or layout falls back through the existing backend path; a direct launcher
call reports a `ValueError`. The model-level TurboQuant FP16 workspace
reservation was not changed; capacity runs must continue to report its peak
alongside candidate allocations.

Initial gfx1201 smoke checks (PyTorch 2.12 / HIP 7.2 container, synthetic
K/V) completed without non-finite output:

| case | block table | dtype | streaming time | raw-current relative L2 |
| --- | --- | --- | ---: | ---: |
| cached 128 / q129 | contiguous | BF16 | 241 us | 0.00223 |
| cached 129 / q129 | permuted physical blocks | BF16 | 272 us | 0.00222 |
| cached 129 / q129 | permuted physical blocks | FP16 | 237 us | 0.000278 |
| cached 255 / q257 | permuted physical blocks | BF16 | 829 us | correctness skipped |

The representative synthetic 32K runs used the separated `--skip-correctness`
mode so old SDPA reference allocation could not gate the candidates. At
32K/q256, runtime-auto old continuation was 81.87 ms and splitless streaming
was 22.09 ms (3.71x); at 32K/q512 they were 154.16 ms and 41.99 ms (3.67x).
The q512 synthetic run completed in this container, so it is not a model OOM
result and is not substituted for the P1 model capacity evidence. Calculated
streaming output workspace was 3.0 MiB/q256 and 6.0 MiB/q512, while the old
explicit accounting was 265.0 MiB and 274.3 MiB respectively; allocator peak
deltas are retained separately in the JSONL records.

### P2.2 real-model validation (2026-09-13)

Validation used revision `5163397b880244656ac427ab3bc14252b2b86607` in the
gfx1201 container (PyTorch `2.12.0+git6bbd260`, HIP `7.2.53211`) with the
official `amd-Qwen3.8-27B-Quark-AWQ-MXFP4` model, TP1, TurboQuant K8/V4,
MTP2, adaptive verification disabled, AITER attention paths disabled, and
`max_num_batched_tokens=256`. The production opt-in was toggled only for the
candidate run; the default remains unchanged. The existing CPU contract tests
passed (`25 passed`), and an additional GPU boundary case with block size 32,
cached/q lengths 129/129, and permuted physical blocks completed with finite
output. That case had raw-current relative L2 `0.002225` and max-abs
`0.003906`.

The real-model 4K smoke requests completed for both paths without errors and
produced the same output hash (`a72956da...`). For cold 32K with chunk 256,
the measured baseline TTFT was `131.988972 s` and the candidate was
`86.218048 s`, a `1.531x` ratio (34.7% lower TTFT), so the cold-prefill speed
gate passes. This is an end-to-end server measurement, not a kernel-only
ratio. The regular output hash nevertheless changed (`6dd87b...` baseline vs
`e2d8de...` candidate), and mean MTP acceptance fell from about `2.826` to
`2.241` (accepted draft rate about 91.3% to 62.1%).

The candidate also completed 32K/chunk512 (`74.820930 s` TTFT) where the P1
baseline OOMed. This is recorded as a capacity/functional win only; no speed
ratio is assigned. A sampled candidate run used about 29.404 GiB of
34.209 GiB physical VRAM, not a guaranteed peak, and the model-level FP16
workspace reservation was not changed. The earlier 64K baseline timeout was
not rerun and no 64K/120K claim is made.

For quality diagnosis, baseline and candidate were run on the same 32K prompt
(prompt hash `e521e829...`) with saved final logits and layer probes. All
captured tensors were finite, but the common first output-position logits had
max-abs `5.0546875`, RMSE `0.859348`, KL `0.628086`, and top-10 overlap 9/10;
the argmax changed from token `52782` to `27775`. At the saved layer-63 probe,
the largest per-position max-abs difference was `29.5` (final-norm probe
`49.0`). These captures are evidence that the real-model numeric/quality gate
has not passed; they are not a model-quality approval. The initial CK-tile
warmup SIGSEGV from an incomplete diagnostic environment was excluded from
the results.

The machine-readable records and diagnostic tensors are preserved under
`/tmp/tq-p2.2-real-20260913/`. P2.2 therefore remains **candidate
implementation only**: the speed and capacity observations are recorded, but
the adoption gate is **not passed** because the production quality evidence
is not acceptable. The current candidate evaluation is closed with the
candidates rejected; it is not a pending request to repeat the same
diagnosis. No threshold, default dispatch, or later P3 work is enabled.

### P2.2 numerical diagnosis before any model re-evaluation (2026-09-13)

A baseline-only capture was taken with the same production model and chunking,
then replayed offline without sending candidate output into any later model
step. The representative snapshot is full-attention layer 63 with query and
raw current K/V shapes `[256,24,256]`, `[256,4,256]`, and a compact SoA cache
`[2029,16,4,388]`; `cached_len=32208` and `seq_len=32464`. The captured
baseline output and the old SDPA replay are bitwise identical.

The candidate's exact cache-load/dequant expression was replayed in a
diagnostic Triton dump. Compared with the old path's FP16 dequant workspace,
both K and V have max-abs zero before and after the FP16-to-BF16 contract
conversion (all elements equal). The cache layout, block mapping, and raw
current tensors therefore do not explain the candidate difference in this
case.

The controlled attention ablations, all using the same Q/K/V and causal
bound, are:

| variant | max-abs vs old | RMSE vs old | relative L2 vs old |
| --- | ---: | ---: | ---: |
| streaming candidate | 0.250000 | 0.00390115 | 0.00076635 |
| online PV FP32 | 0.125000 | 0.00098324 | 0.00019315 |
| online PV FP64 | 0.125000 | 0.00096746 | 0.00019005 |
| online QK + PV FP64 | 0.125000 | 0.00097193 | 0.00019093 |
| tiled FP64 reference | 0.125000 | 0.00097193 | 0.00019093 |

Promoting PV reduces the candidate error by about 75% in RMSE/relative L2,
while promoting QK in addition changes it negligibly. The supported cause
hypothesis is the candidate's PV-side BF16 conversion (`p` and the
BF16-rounded `v`) before the dot product, not cache dequantization or QK.
The proposed fix candidate is to retain the existing FP16-to-BF16 rounding of
cache/current values, but keep the probability and rounded value operands in
FP32 for the PV dot and FP32 accumulation. This remains a default-off
diagnostic candidate; no production threshold, dispatch, or P3 setting is
changed until model quality and speed are re-evaluated.

The replay program and snapshot results are preserved under
`/tmp/tq-p2.2-real-20260913/replaydiag/`; this single-layer diagnosis is a
causal isolation result, not yet a model-quality gate.

### P2.2 PV-FP32 candidate re-evaluation (2026-09-13)

The proposed precision fix was implemented at revision `fc2730b9cd` as a
separate, default-off diagnostic switch:
`VLLM_TQ_GFX1201_K8V4_PREFILL_PV_FP32=true`. It preserves the existing
FP16-to-BF16 rounding of cached values and raw BF16 current values, but keeps
probabilities and those rounded values in FP32 for the PV dot and accumulator.
The normal opt-in, threshold, dispatch, and P3 settings are unchanged.

On the immutable layer-63 snapshot, the exact candidate cache-load replay still
matched the old FP16 workspace K/V at every element. With the switch enabled,
the actual Triton candidate measured max-abs `0.125`, RMSE `0.00100940`, and
relative L2 `0.000198287` against old math SDPA, versus max-abs `0.250`, RMSE
`0.00390115`, and relative L2 `0.000766345` before the fix. The offline
PV-FP32 ablation was therefore reproduced by the kernel, but it remained above
bitwise equality; QK+PV FP64 did not improve the FP64 reference comparison.

The precision change has a substantial cost on the representative synthetic
32K/q256 shape with fixed metadata: the splitless streaming kernel median rose
from `21,239 us` with the original BF16 PV dot to `87,274 us` with the FP32 PV
dot (4.11x slower). This explains why the end-to-end speed gate did not hold.

The same model configuration was then rerun with the switch enabled. The 4K
smoke hash was `c57cd141...`, differing from the baseline/old-candidate hash
`a72956da...`. At 32K/chunk256, the measured TTFT was `159.304263 s`, compared
with `131.988972 s` for the baseline and `86.218048 s` for the original
candidate; the corresponding candidate hash was `d4b23e55...`. The request
record did not contain a client-side speculative-metrics object in this run,
so MTP acceptance is not used as a gate here; the server-side metrics were
retained in the log.

The saved logits/layer diagnostic also did not pass the quality gate. Against
the baseline's common 32K output position, max-abs was `5.9296875`, RMSE
`0.879591`, KL `0.822002`, top-10 overlap `7/10`, and the argmax changed from
`52782` to `760`. The layer-63 probes reached max-abs `34.5` (final-norm probes
`6.625` and `33.0`). These captures had finite tensors but did not establish
model-quality approval; the earlier BF16 candidate had max-abs `5.0546875` and
RMSE `0.859348`, so PV-FP32 does not recover the end-to-end discrepancy.

The artifacts are preserved under `/tmp/tq-p2.2-real-20260913/`, including
`replaydiag/results/numerics-pvfp32.json`, the paired 32K kernel timings, and
`diag/*-pvfp32*`. The PV-FP32 switch is therefore retained only as a recorded
cause-isolation candidate and is **not adopted**. The current P2.2 candidate
evaluation is closed; production defaults, thresholds, and all P3 work remain
unchanged.

### P2.2 saved FP64 comparison and 4K first-difference trace (2026-09-13)

The earlier saved-artifact comparison remains the 32K layer-63 numerical
record. `variants.pt` and `variants-pvfp32.pt` share the old SDPA replay and
the same FP64-contract reference; candidate output was not chained into a
baseline continuation.

| output | max-abs vs FP64 | RMSE vs FP64 | relative L2 vs FP64 |
| --- | ---: | ---: | ---: |
| old SDPA math | 0.125000 | 0.000971927 | 0.000190925 |
| old BF16 streaming candidate | 0.250000 | 0.003898440 | 0.000765808 |
| PV-FP32 streaming candidate | 0.125000 | 0.001319219 | 0.000259147 |

This remains a layer-63 cause-isolation result only. The 4K trace still places
the first output digest difference at `cached_len=256`, `q_len=256`,
`language_model.model.layers.3.self_attn.attn`, with the first raw Q/K/V
digests equal. After the comparator correction, that result is explicitly
`uncompared_input`, because prefix K/V (and, for the old records, scale and
causal-boundary metadata) were not all compared. No attention-operation cause
is claimed from that trace alone. The saved comparison is reproducible with
`benchmarks/kernels/benchmark_turboquant_gfx1201_prefill_saved_compare.py`.

### P2.2 fixed-input layer-3 numerical diagnosis (2026-09-13)

This supplement is limited to one baseline snapshot at the first observed
4K difference: `language_model.model.layers.3.self_attn.attn`,
`cached_len=256`, `q_len=256`, and `seq_len=512`. No 4K/32K variant matrix
was rerun, no candidate output was chained into a baseline continuation, and
no new kernel was implemented. The snapshot was captured at revision
`8fb487f41266db2e9ba634632dc3cf99e26d8704` with Math SDPA explicitly forced.

The snapshot fixes the complete attention contract needed by the offline
comparison: BF16 query and raw current K/V, the old FP16-workspace prefix K/V,
the prefix after the old FP16-to-BF16 conversion, `scale=0.0625`,
`q_positions`, `k_positions`, and the boolean causal mask. The saved metadata
states `sdpa_backend=MATH` and `sdpa_math_forced=true`.

The separate FP64 references are:

| comparison | max-abs | RMSE | relative L2 |
| --- | ---: | ---: | ---: |
| old SDPA vs FP64 pre-cast, BF16-rounded prefix | 0.007798811 | 0.000506864 | 0.001681955 |
| old SDPA vs FP64 then final BF16 cast, BF16-rounded prefix | 0.003906250 | 0.000009112 | 0.000030238 |
| FP64 pre-cast, FP16 prefix -> BF16 prefix input rounding | 0.005733865 | 0.000174260 | 0.000578257 |
| FP64 pre-cast -> final BF16 cast, BF16-rounded prefix | 0.007798811 | 0.000506864 | 0.001681962 |

With the BF16-rounded prefix and the same final BF16 cast, the old SDPA
residual against FP64 is only RMSE `9.11e-6` (relative L2 `3.02e-5`). Removing
the final output cast exposes the much larger RMSE `5.07e-4`; the prefix
FP16-to-BF16 input rounding is recorded separately at RMSE `1.74e-4` before
the final cast. These are fixed-input, single-layer arithmetic observations;
they do not establish a model-wide quality result or prove the candidate's
sub-operation cause.

The trace comparator was corrected at the same time. A first difference is
classified as `attention_operation` only when all required inputs are present
and equal. Existing 4K traces do not contain prefix K/V digests, so their
layer-3 result is now classified as `uncompared_input` rather than an
operation-cause conclusion. Query/raw K/V equality alone is not sufficient.

The CPU/GPU reproduction program is
`benchmarks/kernels/benchmark_turboquant_gfx1201_prefill_snapshot_compare.py`.
The diagnostic hook and corrected trace comparator are
`benchmarks/kernels/p2_2_prefill_trace/sitecustomize.py` and
`benchmarks/kernels/benchmark_turboquant_gfx1201_prefill_trace_compare.py`.
Artifacts are preserved under `/tmp/tq-p2.2-real-20260913/trace4k/`, including
`baseline-layer3-snapshot.pt`,
`baseline-layer3-references.pt`,
`baseline-layer3-numeric-gpu.json`, and
`first-difference-uncompared.json`. Reproduction from the saved snapshot is:

```bash
./.venv/bin/python benchmarks/kernels/benchmark_turboquant_gfx1201_prefill_snapshot_compare.py \
  --snapshot /tmp/tq-p2.2-real-20260913/trace4k/baseline-layer3-snapshot.pt \
  --output /tmp/tq-p2.2-real-20260913/trace4k/baseline-layer3-numeric.json \
  --save-references /tmp/tq-p2.2-real-20260913/trace4k/baseline-layer3-references.pt \
  --device cpu
./.venv/bin/python benchmarks/kernels/benchmark_turboquant_gfx1201_prefill_trace_compare.py \
  --baseline /tmp/tq-p2.2-real-20260913/trace4k/baseline.jsonl \
  --bf16 /tmp/tq-p2.2-real-20260913/trace4k/bf16.jsonl \
  --pvfp32 /tmp/tq-p2.2-real-20260913/trace4k/pvfp32.jsonl \
  --output /tmp/tq-p2.2-real-20260913/trace4k/first-difference-uncompared.json
```

This is a cause-isolation supplement only. Production defaults, thresholds,
P3, and all kernel dispatch remain unchanged.

### P2.2 saved-candidate offline replay completion (2026-09-13)

The final diagnostic supplement re-used the saved layer-3 snapshot and saved
references; it did not capture a new snapshot, rerun the model, or send any
candidate output into a baseline continuation. The replay fixes the complete
contract at `cached_len=256`, `q_len=256`, `scale=0.0625`, explicit Math SDPA,
BF16 query/raw current K/V, BF16-rounded prefix K/V, and the saved causal
boundary. The replay program is
`benchmarks/kernels/benchmark_turboquant_gfx1201_prefill_offline_candidates.py`.

The existing candidate cache reader was run on the saved SoA cache and block
table. Its decoded prefix matched the saved FP16 workspace and the saved
FP16-to-BF16 prefix exactly for both K and V: max-abs and RMSE were zero and
all elements were equal. Prefix dequantization and the cache/block mapping are
therefore not the source of the remaining candidate arithmetic difference in
this fixed case.

Both candidates were replayed offline by a PyTorch implementation of their
precision settings on the same BF16 input. This program does not invoke the
production Triton attention launcher; `--verify-prefix` invokes the existing
Triton cache reader only for prefix-dequantization verification. The rows
below show the final BF16 output against the FP64 reference before and after
the final BF16 cast; the pre-cast candidate rows are retained separately
because the old SDPA artifact only stores its final BF16 output.

| route | max-abs vs FP64 pre-cast | RMSE vs FP64 pre-cast | relative L2 vs FP64 pre-cast | max-abs vs FP64 BF16-cast | RMSE vs FP64 BF16-cast | relative L2 vs FP64 BF16-cast | numeric gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| old SDPA | 0.007798811 | 0.000506864 | 0.001681955 | 0.003906250 | 0.000009112 | 0.000030238 | **pass** |
| BF16 candidate | 0.008922591 | 0.000515869 | 0.001711837 | 0.015625000 | 0.000335916 | 0.001114693 | **fail** |
| PV-FP32 candidate | 0.007798811 | 0.000506864 | 0.001681955 | 0.007812500 | 0.000011382 | 0.000037769 | **fail** |

At the arithmetic stage before the final output cast, the BF16 candidate was
`max-abs=0.002159503`, `RMSE=0.000096773`, and the PV-FP32 candidate was
`max-abs=0.000004222`, `RMSE=0.000000087`, against the BF16-input FP64
reference. The difference exposed after the output cast is therefore recorded
separately from the operation arithmetic.

The existing synthetic numerical gate requires finite candidate output and
both max-abs and RMSE no greater than `1.1x` the old SDPA error against the
same BF16-input FP64 reference after the final BF16 cast. The old SDPA baseline
limits are max-abs `0.004296875` and RMSE `0.000010024`; the BF16 candidate
and PV-FP32 candidate exceed both limits. The fixed-input numeric gate is thus
**not passed** for either candidate. This closes the requested diagnostic
supplement. Both current candidates are rejected and P2.2 evaluation is
finished; no production threshold, dispatch, or P3 work is enabled.

The GPU result, Markdown table, and replay tensors are preserved at
`/tmp/tq-p2.2-real-20260913/trace4k/offline-candidates-gpu.json`,
`offline-candidates-gpu.md`, and `offline-candidate-outputs.pt`. A CPU replay
using the same saved snapshot and references is also recorded as
`offline-candidates.json`. The GPU replay can be reproduced without a model
run with:

```bash
/tmp/tq-venv/bin/python \
  benchmarks/kernels/benchmark_turboquant_gfx1201_prefill_offline_candidates.py \
  --snapshot /cache/tq-p2.2-real-20260913/trace4k/baseline-layer3-snapshot.pt \
  --references /cache/tq-p2.2-real-20260913/trace4k/baseline-layer3-references.pt \
  --output /cache/tq-p2.2-real-20260913/trace4k/offline-candidates.json \
  --markdown-output /cache/tq-p2.2-real-20260913/trace4k/offline-candidates.md \
  --save-outputs /cache/tq-p2.2-real-20260913/trace4k/offline-candidate-outputs.pt \
  --device cuda --verify-prefix
```

### P2.2 evaluation closure (2026-09-13)

The BF16 and PV-FP32 implementations are **evaluated and rejected**. The
diagnostic supplement is complete; no further replay or model run is needed
for these same candidates. The BF16 candidate lacks the required fixed-input
numeric evidence, while PV-FP32 improves the pre-cast local arithmetic but
still fails the final BF16-cast gate and the measured speed requirement.

This is a decision about these two implementations, not a proof that streaming
attention is impossible and not a model-quality conclusion beyond the recorded
gates. Reopening P2.2 requires a different implementation hypothesis and a new
plan/adoption gate. P3 may be evaluated independently against the existing
attention path; it does not depend on enabling or rescuing P2.2. P3 work is now
limited to a default-off implementation and diagnostic benchmark; no adoption
or production dispatch change follows from this start.

## P3: native FP8-WMMA W4A8 large-M only

P3 is independent from failed A3. Small-M calls remain on the current software-fused MXFP4 backend.

The existing `launch_gfx1201_w4a8_prefill()` placeholder is the Python boundary. The functional candidate must use a HIP/native route proven by disassembly to emit the intended gfx1201 FP8 matrix instruction. A Triton route that expands FP8/MXFP4 to BF16 and runs BF16 WMMA is not the target.

The current candidate is implemented in `csrc/rocm/gfx1201_w4a8_prefill.cu`,
registered through `csrc/rocm/ops.h` / `csrc/rocm/torch_bindings.cpp`, and
appended to the ROCm `_rocm_C` target. This keeps the experiment behind the
existing Python boundary and avoids changing the production linear dispatch.

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

### P3 implementation kickoff (2026-09-13)

The candidate exposes separate native quantization and GEMM custom ops. The
quantization op converts BF16 rows to the runtime's native FP8 E4M3 byte
representation and records one FP32 scale per row. The GEMM op consumes those
bytes, decodes group32 MXFP4 weights, and uses a gfx1201 FP8 WMMA tile before
producing BF16 output. The opt-in is default-off and the existing small-M/decode
dispatch is unchanged. A small GPU input probe covered zero, tiny,
saturation, scale-boundary, and non-finite BF16 rows before timing: zero rows
keep scale `1`, finite boundary rows complete, and NaN/Inf rows propagate a NaN
row scale and FP8 NaN bytes. The latter remains a diagnostic behavior, not a
production-quality contract.

Static gfx1201 assembly contains
`v_wmma_f32_16x16x16_fp8_fp8`. Initial GPU probes on seeded random and boundary
inputs matched a reference built from the candidate's actual FP8 bytes
bit-for-bit for the tile-aligned cases. One edge-tile case differed from a Torch
FP32 matmul reference by max-abs `0.5`, while the same output differed from an
FP64 accumulation reference by at most `6.1e-5`; this is treated as a
reduction-order distinction until a production-shaped oracle is applied. Torch's
FP8 conversion also differed on a small number of rounding-boundary bytes in the
same probes (12--26 bytes in the tested cases, with output max-abs differences
up to 20). These are runtime semantic diagnostics, not adoption or production
model-quality decisions.

`benchmarks/kernels/benchmark_gfx1201_w4a8_prefill.py` now records correctness
against the FP64 native-byte oracle and the Torch FP8-conversion reference. It
reports both a pre-dequantized `old_mm_only` diagnostic and an
`old_full_emulation` path that performs production `dequant_mxfp4`, activation
`quant_dequant_mxfp4`, and `F.linear` on every call. The summary applies the
architecture-derived production call weights (64/64/64/48/48/16) separately for
each query-row count. The current candidate timing still uses preallocated
quantization/GEMM workspaces while `old_full_emulation` includes its temporary
allocations, so this is the corrected baseline for the next speed gate but not a
final adoption result until workspace and same-load comparisons are completed.
No cold-32K, quality, or production threshold decision has been made.

### P3 preallocated speed gate probe (2026-09-13)

A Quark-enabled gfx1201 GPU run measured all six tracked dense shapes at
`M=256`, with two warmups, five samples, a 64 MiB device flush, and rotating
operation order. The environment was gfx1201, ROCm `7.2.53211`, and
Torch `2.12.0+git6bbd260`. The native ops came from the P3 source through a
temporary test extension/namespace registration; the production linear dispatch
was not used.
`old_full_emulation` called production `dequant_mxfp4`,
`quant_dequant_mxfp4`, and `F.linear` on every sample. The candidate's combined
operation reused preallocated quantization/output buffers, so this comparison
is favorable to the candidate and still excludes no baseline work.

| N x K | Calls | old_full median (us) | candidate combined (us) | old/candidate |
| ---: | ---: | ---: | ---: | ---: |
| 5120 x 6144 | 64 | 349.8 | 9,045.5 | 0.039x |
| 34816 x 5120 | 64 | 2,152.1 | 52,783.0 | 0.041x |
| 5120 x 17408 | 64 | 1,164.9 | 24,627.4 | 0.047x |
| 16384 x 5120 | 48 | 1,115.9 | 24,454.9 | 0.046x |
| 96 x 5120 | 48 | 94.2 | 1,015.2 | 0.093x |
| 14336 x 5120 | 16 | 930.0 | 21,311.0 | 0.044x |
| **weighted** | **304** | **307,642.6** | **7,096,718.3** | **0.043x** |

The required production-weighted threshold is `1.25x`; the current candidate is
therefore **not adopted** and the speed evaluation is closed for this
implementation. The result is sufficient to stop before workspace reuse,
layout tuning, or cold-32K integration. This is a decision about the current
one-wave row-major implementation, not a proof that a different FP8-WMMA
layout cannot work. Reopening P3 requires a new kernel hypothesis and a new
gate; P4 remains independent and unstarted.

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
4. `[TurboQuant] Stream gfx1201 K8V4 continuation prefill` — P2.2 current candidates evaluated and rejected; evaluation closed. Reopen only with a different implementation hypothesis and a new gate.
5. `[MXFP4] Add gfx1201 native W4A8 prefill` — P3, independent of failed A3.
6. `[TurboQuant] Add gfx1201 raw prefill attention` — P4 only if profile-gated.
7. `[ROCm] Fuse gfx1201 GDN prefill` — P5 only if profile-gated.
8. `[gfx1201] Qualify long-prefill composition` — P6 report/quality/needles/prefix/soak; defaults unchanged.

The P2.2 candidate remains behind its explicit default-off gate and is not enabled. P2.2 evaluation is closed for the current candidates. The current P3 candidate failed its production-weighted speed gate and remains default-off; reopening it requires a new kernel hypothesis. P4/P5 remain unimported and no later production dispatch is enabled.
