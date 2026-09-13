# gfx1201 P3-v2 Radiance W4A8 clean-room design

Status: black-box characterization complete; design only; no kernel or
production integration

Base revision: `f7f3643b00` (`[gfx1201] Link Radiance control summary artifact`)

This document records what can be learned from the pinned Radiance controls
without reading or copying Radiance implementation code.  It turns those
observations into a new, test-only large-tile/multi-wave hypothesis for the
local W4A8 work.  It does not reopen the rejected one-wave 16x16 candidate,
the P2.2 streaming-attention candidate, or any validated production lane.

## 1. Scope and provenance

The controls used the pinned image
`magiccodingman/vllm-radiance@sha256:83a9dc02a8f8e75aabe81366d36ebaa2e35fcbe181cacf8e8e0a4cef4ebccbcc`
with source label `f295b9ef51ad413a68e4192371e0377741a354ce`.  The image was
used as a black-box executable.  No Radiance source, generated source, kernel
body, tile constant, or layout was copied into this repository.  The design
below is a clean-room hypothesis derived from profiler observations and the
existing local W4A8 numerical contract.

The following remain unchanged while this plan is evaluated:

- production defaults and dispatch;
- existing decode, MTP2, FULL_DECODE_ONLY, and K8/V4 cache lanes;
- P2.2 continuation streaming and P3 one-wave FP8-WMMA candidates (both
  remain disabled and closed);
- Radiance provenance and license restrictions.

## 2. Black-box inputs and reproducibility

The three traces are the same cold 32K/chunk256 text control family: TP1,
Quark MXFP4, one request, fixed 64 output tokens, no MTP/speculative decoding,
and the common prompt hash
`0ada0cc394c01ef2450dfca4afaca23149aeb0a365d304732c6bacb06f079dbf`.
The controls were:

| trace | W4A8 | R4D | request TTFT | trace SHA-256 |
| --- | --- | --- | ---: | --- |
| `full` | on | on | 24.825 s | `5bfe599c6cbe363166af1e0779f43b9dd2a532c8d826f4b4ad9f35e563ab907a` |
| `w4a8-off` | off | on | 50.578 s | `b80590729f7eb2cbeb94e5d13ecc3db9be0441daabe028c6a8a75168b79989d7` |
| `r4d-off` | on | off | 26.619 s | `c726c71129155bbe4ef8caa4056e309c934b19686210626e1ca90d5d32c4d593` |

The successful R4D-off control used `--skip-mm-profiling` because the first
attempt failed in an unrelated dummy vision profile.  That startup issue is
not treated as a W4A8 measurement.

The analyzer is [benchmark_gfx1201_radiance_w4a8_characterize.py](../../benchmarks/benchmark_gfx1201_radiance_w4a8_characterize.py).
It streams the trace, assigns a kernel to the smallest enclosing
`execute_context_*` annotation by timestamp midpoint, and saves raw count and
latency distributions.  Example reproduction commands are:

```text
.venv/bin/python benchmarks/benchmark_gfx1201_radiance_w4a8_characterize.py \
  --trace /tmp/tq-radiance-cache-20260913/full/profiler/rank0.1789322729653085518.pt.trace.json.gz \
  --output /tmp/tq-radiance-delta-20260913/radiance-controls/w4a8-characterization/full.json
```

The corresponding offline reports are retained at
`/tmp/tq-radiance-delta-20260913/radiance-controls/w4a8-characterization/{full,w4a8-off,r4d-off}.json`.
Durations in the tables below are sums of kernel durations; overlapping GPU
streams mean they are not a serialized wall-clock critical path.

## 3. Observed execution structure

### 3.1 Context inventory

The full trace contains 143 prefill contexts and 63 decode contexts.  Prefill
is `1 x q256` first chunk followed by `122 x q256 + 20 x q64` continuation
contexts; the annotated prefill token sum is exactly 32,768.  Decode contexts
are annotated with `q_len=0` by this trace format.

The exact native folded W4A8 kernel name is:

```text
void radiance_mxfp4_fp8_gemm_folded<2, true, true>(
    unsigned char const*, unsigned char const*, unsigned char const*,
    unsigned char const*, float const*, std::bfloat16_t*, int, int, int)
    [clone .kd]
```

This is an observation about the profiler name only.  The three integer
arguments and the pointer roles are intentionally not renamed or interpreted.
The trace event arguments contain `device`, `stream`, `correlation`, and
`kind`, but no grid, block, wave-count, VGPR, LDS, scratch, or ISA metadata.

### 3.2 Native folded-kernel measurements

The full-control observations for the exact `...gemm_folded<2,true,true>...`
name are:

| phase | annotated q_len | calls | kernel sum (ms) | p50 (us) | p95 (us) | max (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| first chunk | 256 | 256 | 77.570 | 305.4 | 483.1 | 1,649.9 |
| continuation | 256 | 31,232 | 8,591.083 | 275.9 | 462.6 | 527.9 |
| continuation | 64 | 1,280 | 511.600 | 398.7 | 412.1 | 633.0 |
| decode | 0 | 4,032 | 1,534.098 | 381.4 | 389.2 | 397.6 |

The prefill folded-kernel total is therefore `32,768 calls / 9,180.253 ms`; its call count
equals the annotated prefill token sum, and each individual full-control
prefill context has a folded-call count equal to its q_len.  This establishes a
repeatable trace property, not the semantic meaning of one kernel call.

The q64 and q256 continuation calls have visibly different per-call latency
distributions despite using the same folded kernel name.  The first chunk also
contains a long-tail outlier.  Any clean-room mapping must measure q64 and q256
separately rather than extrapolating one from the other.

The trace also contains separate native decode-family names, including
`radiance_mxfp4_fp8_gemm_decode<8,128,4,1,true,true,true>` and
`radiance_mxfp4_fp8_gemm_decode<8,128,1,1,true,true,true>`.  Their presence is
evidence that the prefill folded family and the decode family are distinct
black-box entry points.  P3-v2 must not alter the decode family.

The full trace has 52,736 calls to
`dynamic_per_token_scaled_fp8_quant_kernel_strided`.  This is evidence of a
separate activation-quantization producer in the observed execution; it does
not prove which folded-kernel argument consumes each producer result.

### 3.3 W4A8-off decomposition

With W4A8 disabled, the trace exposes separate fallback families.  Summed over
prefill contexts, the selected names are:

| observed family | calls | kernel sum (ms) |
| --- | ---: | ---: |
| `dq_uint8_mxfp4_to_half_kernel` | 43,472 | 13,376.419 |
| `qdq_mxfp4_kernel` | 43,472 | 2,584.233 |
| `Cijk_*` dense GEMM | 43,472 | 21,223.586 |

The fallback trace also exposes Cijk names containing strings such as
`MT128x128x32`, `MT64x64x64`, and `MT16x32x256`.  Those names describe the
fallback kernel family only; they are not evidence of the native Radiance
folded kernel's tile or wave shape.

The full request controls are consistent with this decomposition: W4A8-off
raised TTFT from 24.825 s to 50.578 s and raised the classified prefill linear
sum from 11,521.0 ms to 37,213.6 ms.  R4D-off retained the native folded name;
its prefill folded sum was 9,471.567 ms, but its call population differed
(`304` first-chunk calls and `37,088` q256-continuation calls), so it is a
control for composition, not a tile-performance A/B.

## 4. What the observations do and do not establish

Established:

- a native folded W4A8 entry point is present in the executable and is used in
  both prefill and decode contexts;
- the native entry point has packed-byte, float-scale, BF16-output pointer
  types in its profiler signature;
- native folded prefill call counts and latency distributions change with the
  annotated q_len;
- disabling W4A8 exposes separate MXFP4 dequantization, activation QDQ, and
  dense GEMM families and substantially increases prefill cost;
- the native and fallback controls have different output hashes, so these are
  composition controls rather than quality-equivalent runs.

Not established:

- native grid/block dimensions, wave count, tile dimensions, LDS use,
  register pressure, memory transaction pattern, or generated ISA;
- whether a folded call processes one token, a token tile, or another packed
  work unit;
- that the native kernel decodes a weight tile once per workgroup, uses a
  particular fragment permutation, or applies a particular staging schedule;
- numerical quality of the black-box output from a single greedy hash;
- that any Radiance implementation detail is licensed for source reuse.

The missing launch metadata is why P3-v2 does not claim to reproduce a
Radiance tile.  It proposes a measurable local mapping instead.

## 5. P3-v2 clean-room hypothesis

The hypothesis is deliberately narrower than “rewrite attention” or “port
Radiance.”  It is a benchmark-only large-M replacement for the rejected local
one-wave 16x16 FP8-WMMA mapping, while preserving the local MXFP4/E8M0 and
dynamic-FP8 activation contract already used by the P3 prototype.

### 5.1 Work partition

Start with a 2-wave and 4-wave workgroup sweep.  The first tile candidates are
`64x64`, `64x128`, and `128x64` in `(M,N)`; `128x128` is allowed only if the
measured LDS/register footprint leaves usable occupancy.  These are proposed
test points, not observed Radiance constants.

For each workgroup:

1. load one packed MXFP4 weight tile and its local E8M0 scale metadata;
2. decode each packed byte once into an FP8 tile in LDS/register staging;
3. let multiple waves consume that tile for distinct M/N output tiles;
4. keep FP32 accumulators in registers across the K loop, avoiding the
   per-K LDS accumulator store/read/modify/write pattern of the rejected
   one-wave candidate;
5. apply the existing group-scale rule at the K-group boundary, then continue
   the same accumulator; and
6. write the BF16 output once after the K reduction is complete.

Double-buffered LDS staging is a tunable option, not a requirement.  A split-K
variant is evaluated only when the M/N tile does not provide enough parallel
work; its reduction cost must be reported separately.  No decode, attention,
KV layout, MTP, or scheduler code belongs in this experiment.

### 5.2 Why this follows from the observations

The trace shows a fused native family is valuable relative to the exposed
dequant-plus-Cijk fallback, but it does not show how that family is tiled.  The
local P3 candidate's measured one-wave/16x16 mapping was rejected because its
kernel-only throughput collapsed despite emitting native FP8 WMMA.  Therefore
the next clean-room test should change only the work partition and reuse
schedule: more than one wave per output tile, larger M/N reuse, and no
intermediate accumulator spill.  This is a falsifiable hypothesis, not a claim
about Radiance internals.

### 5.3 Measurement matrix

The future prototype must remain an offline microbenchmark with preallocated
inputs, output, and workspace.  It should rotate A/B order, warm up before
sampling, save raw samples, and record compiler/ISA metadata whenever the
local toolchain exposes it.  At minimum use the already recorded large-M
production shapes and both `M=256` and the observed `M=64` continuation shape.
Measure separately:

- activation quantization producer;
- packed-weight decode/staging;
- WMMA/GEMM body;
- scale application and output cast;
- combined candidate time.

This decomposition prevents a kernel-only win from being mistaken for a
production full-emulation win.

## 6. Adoption gates and stop conditions

P3-v2 is not eligible for production from this document.  If a prototype is
later implemented, it must pass all of these in order:

1. **Numerical gate:** finite output and the existing P3 FP64-byte-oracle
   thresholds (`max-abs <= 0.004296875` and the recorded RMSE limit of about
   `1.0024e-5`) on actual native FP8 bytes, including K-group tails and M/N
   tails.  Cast-before and final-BF16 comparisons must both be retained.
2. **Kernel gate:** at least `1.25x` the local `old_full_emulation` production
   call-weighted baseline at representative large-M shapes, with raw samples
   and no allocation or compilation time in the kernel-only number.
3. **Shape gate:** no correctness failure at `M=64`, `M=256`, and the selected
   large-M shapes; q64 and q256 are reported independently.
4. **Integration gate:** only after gates 1–3 may a separate same-process cold
   32K model A/B be proposed.  That A/B requires quality and capacity review;
   it is not part of P3-v2 characterization.

Failure at any gate closes that candidate.  It does not reopen P2.2, the
one-wave P3 candidate, P4 attention, P5 GDN, or the production defaults.

## 7. Current decision

The black-box characterization is complete and the large-tile/multi-wave
clean-room hypothesis is recorded.  No P3-v2 kernel has been written, no
production launcher or threshold has changed, and no Radiance source has been
copied.  The next authorized action is a separately reviewed benchmark-only
prototype; until then the validated baseline remains the rollback target.
