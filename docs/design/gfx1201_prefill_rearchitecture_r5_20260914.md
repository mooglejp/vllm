# R5 existing AMD Triton attention diagnostic

Date: 2026-09-14  
Start commit: `c8b800ef786230644ac29bb0a3bd9b1b9c904a5f`  
Worktree branch: `codex/gfx1201-prefill-r5`

## Scope and stop rule

This is an independent R5 diagnostic. R1/H2 remains rejected and preserved;
R2, MXFP4 fusion, P2.2 continuation streaming, production dispatch, decode,
MTP2, and KV layout were not changed. No new attention kernel was written.

The test used the public AMD Triton FlashAttention entry point only. Timing and
model A/B were conditional on every numerical case passing. Because the fixed
numerical gate failed, the run stops before attention timing, continuation
timing, or production integration.

## Backend provenance

The isolated verification process loaded `flash_attn` 2.8.3 from the pinned
wheel `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` (wheel SHA-256
`af2d6326747cd49a43843d04d01ebcdf3b452ca3f873b7435a21d140ff598b1a`). The
wheel did not expose an upstream source commit. The selector was
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, and the imported interface reported
`USE_TRITON_ROCM=True`.

The public entry point was `flash_attn.flash_attn_varlen_func`; the loaded AMD
files and hashes are recorded in the machine-readable artifact. `aiter` was
not used. The trace captured `attn_fwd.kd` under
`flash_attn::_flash_attn_varlen_forward`; this was not CK, Math SDPA, or an
unverified fallback. The run used an AMD Radeon AI PRO R9700 (`gfx1201`),
Torch `2.12.0+rocm7.14.0`, and HIP `7.14.60850`.

## Fixed numerical contract

All cases used BF16, `D=256`, `Hq=24`, `Hk=4`, GQA6, one request, dropout zero,
no sinks, and no sliding window. Prefix K/V were written to the existing SoA
K8/V4 cache, read by the existing reader into FP16, and converted to BF16.
The current chunk remained raw BF16 K/V; it was not requantized. The scale was
`0.0625`. A query row `i` could read exactly `j <= cached_len + i`.

The fixed `cached_len=3, q_len=2` fixture produced the required bottom-right
mask:

```text
1 1 1 1 0
1 1 1 1 1
```

Both explicit Math SDPA and the AMD Triton backend returned `[1.5, 2.0]` for
the fixture. The required q lengths were tested for prefix lengths 0, 3, and
4096; the additional long case was 32768/q256. Small cases used the full
output oracle. Long cases used first/middle/last rows and heads 0/12/23 while
retaining the original absolute query positions.

The independent oracle computes the selected rows in FP64. Both final BF16
values and pre-cast FP64 values were saved with max-abs, RMSE, relative L2,
finite, and mismatch counts. The upstream AMD Triton test source in the loaded
wheel fixes `torch.testing.assert_close` at `atol=1e-2`, `rtol=1e-2`; these
values were recorded before measurement and were not changed after seeing the
results. The candidate passed that broad upstream check, but that is not the
R5 equivalence gate.

The R5 candidate gate was fixed as candidate final-BF16 max-abs and RMSE no
more than 1.10 times the same-input explicit Math SDPA baseline, with finite
outputs. This intentionally does not reuse a P2.2 or GEMM absolute threshold.

## Result

The boundary fixture passed, and all captured backend outputs were finite. The
candidate nevertheless failed the fixed equivalence gate in 18 of 19 requested
cases (only cached=0/q=1 passed). Representative results are:

| case | Math max-abs / RMSE | AMD Triton max-abs / RMSE | gate |
| --- | ---: | ---: | --- |
| cached=0, q=127 | 0.00390625 / 7.63e-6 | 0.015625 / 5.32e-4 | fail |
| cached=3, q=256 | 0 / 0 | 0.0078125 / 6.92e-4 | fail |
| cached=4096, q=256 | 0 / 0 | 4.88e-4 / 6.81e-5 | fail |
| cached=32768, q=256 | 7.63e-6 / 1.59e-7 | 1.22e-4 / 2.28e-5 | fail |

For zero-baseline cases, the fixed 1.10x rule correctly gives a zero allowance;
finite nonzero candidate error is therefore a failure rather than a reason to
relax the gate. Since at least one required case failed, no timing was taken:
each case records `not_measured_until_all_numerical_cases_pass`. Consequently
the 1.3x continuation gate, the 4K/32K continuation comparison, the 10% cold
32K TTFT gate, and model A/B were not attempted.

## Artifacts and decision

The complete machine-readable report, including raw per-case numerical metrics,
backend hashes, boundary fixture, and the `attn_fwd.kd` trace, is:

`docs/design/artifacts/gfx1201_prefill_rearchitecture_r5_20260914.json`

The benchmark-only reproducer is:

`benchmarks/kernels/benchmark_gfx1201_r5_attention.py`

**Decision: numerical gate failed; stop.** The existing AMD Triton backend is
not qualified for K8/V4 large-continuation replacement under this fixed
contract. This result does not reject K8/V4 storage or AMD Triton attention in
general, and it does not authorize an independent precision workaround or a
new kernel. Any future R5 restart requires a separately reviewed numerical
hypothesis and gates.
