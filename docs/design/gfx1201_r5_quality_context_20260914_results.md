# R5 context-qualified quality evaluation — result

The fixed additional evaluation ran once from preparation commit
`adf964f532866f80c75e03807e38d4f59460e742`. It used one model process and 616
sequential requests (308 baseline, 308 candidate), with the candidate enabled
only for the existing target continuation consumer. No attention arithmetic,
input contract, KV layout, production dispatch, default, decode path, or MTP2
setting was changed.

The result is **additional quality gate failed; stop before 32K retention**.
The earlier 308-case stop, HumanEval/129 observation, strict Math-ratio failure,
and cold-32K TTFT 2.1733x result remain unchanged historical results.

## Frozen input and environment

The generated suite has SHA-256
`0df298d4bf625a93246ada772c07fe4e5b721eb2f26ccedddc043bf340bda1de` and every
prompt has exactly 2,048 token IDs. Output caps were GSM8K 1,024, MMLU 64, and
HumanEval 2,048. The source suite, construction checks, insertion positions,
sampling, judge hashes, and gate rules are in
`gfx1201_r5_quality_context_manifest_20260914.json`.

Both arms used the same prompt token IDs and output caps per case. Baseline had
zero candidate applications. Candidate application and problem-overlap
coverage were both 308/308; the paired target shapes matched 308/308. The
repaired interactive HumanEval controls passed before grading, and both arms
had all 164 isolated judgments. No request returned an empty output or a
non-finite token log probability.

The environment remained TP1, TurboQuant K8/V4, MTP2, adaptive verification
off, fused MXFP4 decode on, compilation mode 0, `FULL_DECODE_ONLY`, chunk budget
256, `max_num_seqs=1`, RAM limit 16 GiB, swap disabled, no profiler, and
disk-backed artifacts.

## Fixed paired result

| task | baseline correct | candidate correct | baseline length stops | candidate length stops | completed-pair format invalid (B/C) | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GSM8K (64) | 36 | 35 | 0 | 0 | 0 / 0 | fail: score decreased |
| MMLU (80) | 65 | 65 | 0 | 0 | 0 / 0 | pass |
| HumanEval (164) | 138 | 139 | 1 | 2 | 2 / 1 | fail: one extra stop and a new invalid |

The baseline length stop was `HumanEval/147`. Candidate length stops were
`HumanEval/32` and `HumanEval/147`. The completed-pair syntax/extraction gate
found one new candidate-invalid case, `HumanEval/93`, and two resolved cases,
`HumanEval/10` and `HumanEval/129`. Raw Markdown violations are recorded
separately (46 baseline, 50 candidate for completed HumanEval pairs); they are
not substituted for the extracted-source syntax result.

The candidate's HumanEval functional count being one higher does not offset the
GSM8K decrease, the extra HumanEval truncation, or the new completed-pair
format-invalid case. No answer was repaired, excluded, or regenerated.

## Coverage and resources

| observation | baseline | candidate |
| --- | ---: | ---: |
| requests | 308 | 308 |
| candidate-applied requests | 0 | 308 |
| applied calls | 0 | 34,496 |
| applied calls overlapping problem | 0 | 8,672 |
| output tokens | 24,745 | 28,221 |
| total request elapsed | 1,771.985 s | 1,793.327 s |
| median request elapsed | 4.391 s | 4.145 s |
| p95 request elapsed | 10.751 s | 11.204 s |
| sampled response RAM peak | 15,715,729,408 B | 15,715,028,992 B |
| process-lifetime GPU allocator peak allocated/reserved | 28,868,519,936 / 29,223,813,120 B | same |

The cgroup OOM, OOM-kill, and group-kill deltas were all zero. After shutdown,
the container reported 9,656,053,760 B RAM, `/dev/shm` used 2,213,326,848 B,
and `rocm-smi` reported 59,912,192 B VRAM used. These are post-shutdown
observations; no continuous tmpfs profiler was run, so they are not runtime
peaks.

## Stopping point and artifacts

Because the additional quality gate failed, 32K needle/content retention was
not run. Decode regression, prefix reuse, soak, 64K/120K, production
registration, and default enablement were not run. The candidate remains
diagnostic-only and no follow-up kernel or backend search was started.

The complete raw archive is
`gfx1201_r5_quality_context_20260914.tar.gz`, SHA-256
`51d22a69531497ba460b8bd6977461aa78d5496f4c4c6ee1258f542da6a70926`. It
contains both 42 MiB response JSONL files, raw outputs, isolated HumanEval
judgments, the report, environment, resource summary, suite, and server log.
The machine-readable result is
`gfx1201_r5_quality_context_result_20260914.json`; no earlier artifact is
overwritten.
