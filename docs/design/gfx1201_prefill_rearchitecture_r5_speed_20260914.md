# R5 evaluation policy v2: upstream tolerance and continuation timing

Start: `66cd1152ab`, isolated branch `codex/gfx1201-prefill-r5`.
The original checkout's user modification to `.gitignore` is preserved.

## Policy fixed before this measurement

The [strict evaluation](gfx1201_prefill_rearchitecture_r5_20260914.md) and its
artifact remain unchanged. Its Math-relative 1.10x gate failed. That result is
a strict reproducibility diagnostic, not rejection of the AMD Triton backend.

This separate policy permits timing after the causal fixture, input contract,
full-output finiteness and upstream-derived tolerance pass. The loaded
FlashAttention 2.8.3 `flash_attn_triton_amd/test.py` uses
`torch.testing.assert_close`, `atol=0.01`, `rtol=0.01`, against
`attention_forward_pytorch_ref_impl`, with FP16 inputs. Here the same formula
`abs(actual-reference) <= 0.01 + 0.01 * abs(reference)` is applied to BF16
candidate versus explicit Math SDPA over identical effective BF16 inputs.
This is a documented BF16/reference adaptation, not a claim to reproduce the
upstream FP16 test exactly. Full-output finiteness is independently required.
The source hashes and precise package paths are in the artifact.

Both paths use the same Q, retained SoA K8/V4 cache decoded through FP16 then
BF16, raw BF16 current K/V, scale 0.0625, and bottom-right causal boundary.
The original boundary suite remains preserved; this run rechecks the fixed
cached3/q2 fixture and the two long timing cases. Inputs are synthetic, not
captured model activations. FP64 reference rows preserve absolute positions.
Differences and reductions are calculated in FP64. The names
`fp64_reference_difference` and `final_bf16_reference_difference` distinguish
the independent unrounded oracle from its BF16-rounded value. Both compare
the kernel's **final BF16 output**; neither observes its internal accumulator.

## Timing boundaries and results

Cache allocation/store and fixed metadata are prepared before timing.
Attention-only includes the public attention API and its output allocations.
Continuation includes decoding existing cache, FP16-to-BF16 conversion,
joining raw current K/V, required mask or cumulative-length metadata, output
allocation and attention. No explicit preparation synchronization remains
inside that interval. Both arms use five warmups, 20 raw samples, alternating
order and a 64 MiB flush. The measurement uses ROCm events rather than CUDA
CUPTI, following the existing ROCm harness.

| Prefix / query | Math continuation median | AMD Triton median | Speedup |
| --- | ---: | ---: | ---: |
| 4096 / 256 | 4.449870 ms | 0.918547 ms | 4.844x |
| 32768 / 256 | 38.099447 ms | 5.261342 ms | 7.241x |

Both cases pass the v2 numerical prerequisites and fail the retained strict
Math-ratio diagnostic. The 32K continuation threshold of 1.30x passes.
These are fixed-input continuation measurements, not model TTFT speedups.
The same R9700 uses Torch 2.12.0+rocm7.14.0 / HIP 7.14.60850, the pinned
FlashAttention 2.8.3 wheel and AMD selector from the strict record.

## Model A/B validation

**Stopped: model A/B blocked by host RAM and bounded-container startup OOM.**
The 10% cold32K TTFT threshold is unchanged and is **not evaluated**. There are
zero valid 32K timing samples. This is neither a TTFT gate failure nor an AMD
Triton numerical rejection. Quality, prefix reuse and operational qualification
remain pending; no path is adopted.

The existing verification container uses Torch `2.12.0+git6bbd260` /
HIP `7.2.53211`, unlike the standalone ROCm 7.14 environment above. Before
the bounded model retry, the same numerical and timing harness was therefore
also executed in this model environment. Both numerical cases passed v2 and
failed strict Math equivalence. Its separate timings were:

| Prefix / query | Math continuation median | AMD Triton median | Speedup |
| --- | ---: | ---: | ---: |
| 4096 / 256 | 9.408469 ms | 0.954278 ms | 9.859x |
| 32768 / 256 | 79.801796 ms | 5.434634 ms | 14.684x |

These values are not pooled with the ROCm 7.14 samples. Each comparison is
same-input and same-environment. The substantial Math latency difference
between environments is observed, not attributed to any one runtime change.
See [model-environment replay](artifacts/gfx1201_prefill_rearchitecture_r5_modelenv_speed_20260914.json).

The diagnostic hook uses the existing materialization unchanged. Both arms
restore the previous diagnostic Math baseline by disabling the unsupported CK
prefill consumer and generic kernel warmup, as in the earlier P2.2 diagnostic
setup. Only target `language_model.model.layers.*`, cached length >0 and query
length >128 can temporarily select the public AMD Triton consumer. First chunk,
short continuation and MTP keep the common baseline behavior. A control file
permits serialized same-process A/B; per-request counts and shape/layer records
verify the consumer switch. This adds equal bookkeeping in both arms.

The validated flags include software-fused MXFP4 decode, K8/V4, MTP2 with
adaptive verification disabled, compilation mode 0, FULL_DECODE_ONLY, one
sequence, 256-token chunk budget and 64 output tokens ignoring EOS. The
container mounts the original checkout at `9938409e92` read-only. Comparing
its `vllm/` tree to `66cd1152ab` shows only the unused R1 `gfx1201_prefill_v3.py`
addition; the attention and validated decode sources are identical.

Both 4K smoke arms generated 64 tokens. Each saw 240 eligible calls across the
16 target attention layers; baseline applied the candidate zero times and the
candidate applied it 240 times. Actual chunk shapes include page-alignment
tails (for example cached3888/q192), while the configured chunk budget stays
256. The supplied token-ID prompt hashes match. Baseline smoke TTFT was
7.404493 s; candidate smoke was 6.179455 s **with the profiler**, so these are
not a speed comparison or a TTFT gate result. Token differences are preserved
without declaring a quality failure.

## Host RAM incident and execution stop

The initial model launcher incorrectly relied on `TQ_DISABLE_FLASH_PREFILL`,
which the current production source does not read; it consequently reached
the known CK startup failure. The common diagnostic Math hook restored the
intended baseline. Earlier exploratory smoke attempts also lacked the
software-fused MXFP4 flag and are excluded from the reported model smoke.

The v6 profiler configuration inherited `torch_profiler_with_stack=True` and
included frontend tracing. The trace capture/export/teardown was followed by
a stop-profile RPC timeout. At **09:16:47 UTC**, the kernel recorded a **global
host OOM**, killing `VLLM::EngineCor` with 12,783,016 KiB anonymous RSS; free
swap was 168 KiB. Retained diagnostic API processes consumed additional RAM.
This coincided with the user's SSH disruption. The log proves system memory
exhaustion, but does not isolate every component's contribution or establish
the exact SSH disconnect cause. This was not a reported GPU-VRAM OOM.

The subsequent baseline 32K warmup returned no generated tokens, no TTFT and
no target scope record. Its 200-second elapsed time is an aborted request,
not a baseline latency. The server and request artifacts are preserved.

All remaining diagnostic processes were stopped. The otherwise idle validation
container was then constrained to **12 GiB RAM and zero swap**. Without running
the profiler, its next model load reached that bound at **09:22:30 UTC** and
the kernel recorded `CONSTRAINT_MEMCG` OOM, with the cgroup kill counter rising
from 8 to 9. The limit was not raised and no further model retry was made.
After cleanup the host reported approximately 19 GiB available RAM. The
diagnostic container retains the bounded resource setting and no model process.

The launcher now fails closed without a <=12 GiB cgroup limit and swap disabled.
Its future connection trace is limited to four iterations, without Python
stacks or frontend capture. No such revised GPU trace was attempted after the
bounded startup failure. Restart requires a model verification environment
that fits the bounded RAM budget, not further kernel changes or an automatic
increase in the memory limit.

The [status manifest](artifacts/gfx1201_r5_model_status_20260914.json),
[kernel OOM excerpts](artifacts/gfx1201_r5_host_oom_20260914.log),
[v6 server log](artifacts/gfx1201_r5_model_v6_20260914.log.gz), and
[bounded v7 server log](artifacts/gfx1201_r5_model_v7_20260914.log.gz)
preserve the stop evidence. The v6 compressed GPU trace passes gzip integrity
checking and is analyzed with a streaming reader rather than loaded wholesale
into host RAM. Its timings do not enter any gate.

The [streamed trace summary](artifacts/gfx1201_r5_model_trace_summary_20260914.json)
maps 240 CPU candidate scopes through launch correlation to 240 `attn_fwd.kd`
kernels and 240 existing prefix-dequant kernels, with no correlated CK kernel.
GPU annotation copies are excluded from the CPU scope count. The analyzer runs
under a 512 MiB address-space limit and does not load the entire trace into RAM.

## Reproduction

Run the following in the same pinned environment, with the isolated
FlashAttention wheel directory on `PYTHONPATH`:

```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  .venv/bin/python benchmarks/kernels/benchmark_gfx1201_r5_attention_speed.py \
  --output /tmp/r5-attention-speed.json --warmups 5 --samples 20 --flush-mib 64
```

The [machine-readable artifact](artifacts/gfx1201_prefill_rearchitecture_r5_speed_20260914.json)
contains every raw timing sample, full numerical checks, independent FP64
metrics, backend provenance and the policy manifest. No production source,
kernel, cache format or rejected candidate was changed.

Model reproduction files are `benchmarks/launch_gfx1201_r5_model_ab.sh`,
`benchmarks/r5_model_ab/sitecustomize.py`,
`benchmarks/benchmark_gfx1201_r5_model_ab_hook.py`, and the model client and
five-pair runner in `benchmarks/`. Install them only into a dedicated verification
environment with the pinned AMD wheel and explicit resource limits. The exact
[prompt token IDs](artifacts/gfx1201_r5_prompt_tokens_20260914.json) are saved.
The runner performs untimed 32K warmups followed by five alternating pairs;
it stops on incomplete generation or unexpected scope counts. That matrix did
not run to completion here. The report aggregator refuses missing samples.

Validation: standalone and model-environment GPU numerical/timing checks;
4K baseline/candidate generation and scope checks; CPU FP64-reduction precision
and raw-sample/median consistency checks; source formatting/lint checks. Model
TTFT, quality, prefix reuse and soak validation remain incomplete.

The two CPU tests in `tests/benchmarks/test_gfx1201_r5_reports.py` pass. They
check that GPU annotation copies cannot double the candidate scope count,
outside-scope kernels cannot be attributed to the candidate, and a truncated
trace array header fails instead of looping indefinitely.
