# R5 quality and operations qualification

## Scope and immutable results

Base: `d5fec6675f832ba6bf753247acd092ef86e5cd56`, isolated worktree
`/tmp/vllm-tq-prefill-r5`, branch `codex/gfx1201-prefill-r5`.
The worktree was clean; the original checkout's modified `.gitignore` was
preserved. No kernel, attention input contract, dispatch scope, production
default, or previously rejected candidate is changed.

The previous cold32K TTFT pass (131.796047 versus 60.643420 seconds,
2.1733x) and the strict Math-ratio numerical diagnostic failure are separate,
unchanged results. This task does not adopt the backend even if qualification
passes. R1/H2, R2, MXFP4 fusion and larger context/chunk experiments remain
out of scope.

## Fixed environment and input

Use the same existing `tq-e2e-current` environment as the TTFT pass:
Torch `2.12.0+git6bbd260`, HIP `7.2.53211`, pinned FlashAttention 2.8.3
with official AMD Triton selection. Model config, tokenizer config, package
record, launcher and injected hook hashes are captured in `environment.json`.
Both arms retain TP1, TurboQuant K8/V4, MTP2, adaptive verification disabled,
validated fused MXFP4 decode, compilation mode 0, FULL_DECODE_ONLY,
chunk budget 256 and max_num_seqs 1.
The model remains the read-only
`/srv/ai/models/llm/hf/amd-Qwen3.8-27B-Quark-AWQ-MXFP4` mount. The unchanged
runtime source mount is the original checkout at `9938409e924b`, as in the
previous successful model environment; the diagnostic worktree is separate.

Only one model server is loaded. RAM is capped at 16 GiB, swap is disabled,
and no profiler is started. Existing compiler caches are retained, not
rebuilt into a new SDK. Logs, responses and resource samples are written to
the disk-backed `/cache/r5-quality-20260914` mount, not tmpfs. The resource
watcher samples cgroup RAM/events, host meminfo, VRAM sysfs and tmpfs usage
every ten seconds and terminates this diagnostic server on a new OOM.

The original suite was located at
`/tmp/tq-final-validation.Awe2nr/suite-308.jsonl`. SHA256 matches exactly:
`8476e86f75bc2b08a19f187e6ca11bef4af161bed203a0817c39a50b1150338a`.
It contains GSM8K 64, HumanEval 164 and MMLU 80 cases. Stored token-ID prompts
already contain the original non-thinking chat template. No questions or
references are regenerated. Generation uses seed 1201, temperature 0,
top_p 1, top_k -1, normal EOS stopping, no added special tokens, and the
original task limits 128/384/16 respectively. The generation manifest is
written before the first request. Returned token log probabilities are checked
for finiteness; this is not exhaustive hidden-activation validation.

## Diagnostic safety and preparation issues

The hook now resets counters on attention `forward`, including first chunks
with no eligible continuation. It rejects changing mode within a run ID.
The attention computation and target-only `cached_len>0 && q_len>128` scope
are unchanged. The client waits for zero running and waiting requests before
switching mode, writes control atomically, and executes one request at a time.
Order alternates B/C and C/B by case. Cache salts encode the unique
trial/mode/case identity as a SHA256 digest; arms do not share cache entries.
Zero-application requests are explicitly marked `unapplied_regression`.

Before evaluation, transfer verification found that a container copy still
contained the old hook. Startup was stopped before any quality request;
the file was copied through the bind-mounted disk and its hash verified.
The initial API request then returned HTTP 400 before generation because
the trial/case identifier contained `/`, which cache-salt validation forbids.
The salt representation was corrected to a digest, without changing any
prompt, sampling option, scoring rule or quality gate. No generated answer
was retried. Both preparation attempts and server logs are retained.

## Scoring and stopping rules

Existing `score.py` and `judge_humaneval.py` are retained in the raw archive;
reviewable copies under `benchmarks/r5_quality` follow repository headers and
regex import style. The scoring expressions and source extraction rules are
unchanged. The isolated judge additionally disables swap; on the existing
ten-second client timeout it kills only its uniquely named container.
It uses the existing image by digest, a read-only filesystem, non-root UID,
no capabilities/network, no-new-privileges, bounded tmpfs, 512 MiB RAM,
one CPU and 64 PIDs. Generated code is never executed on the host.

**The original judge did not execute its input.** It invoked Docker without
`--interactive` while sending the test script on stdin. Python therefore read
EOF and exited successfully. A known-failing control was incorrectly reported
as passed. The corrected judge opens stdin; positive/negative controls now
pass/fail as expected. Both invalid 164/164 results from the first grading
attempt are retained with `invalid-no-stdin` filenames. The original helper
has the same defect, so historical 164/164 claims from that helper are not
valid functional-execution evidence. Historical records are not overwritten.
The saved model answers were graded again with stdin connected; no answers
were regenerated and no test, extraction rule or timeout was relaxed.

The repository regex-import adaptation was checked against the original
helpers on both the 616 historical and 616 current answers. Semantic scoring
and code extraction agree exactly. This equivalence check did not execute
generated code or validate the old Docker invocation.

Require per-task correct counts not below the paired MTP2 baseline, with no
new generation errors. Truncation, empty output, parse failures and coverage
are reported separately. Token/hash disagreement alone does not fail quality.
No old target-only score is used as this baseline. A failed quality gate ends
the experiment without decode/prefix/soak or any precision tuning.

## Fixed-state decode availability

The existing `.tools/tq_greedy_diag/target_replay.py` explicitly omits the MTP
model and requires graphs disabled. It is not a same-state target/drafter
decode replay for this configuration. `test_gpu_trace_replay.py` exercises
forced sampled tokens, not restoration of KV, GDN and MTP states. No valid
full-state harness has been identified; natural generation cannot substitute
for the fixed-state 2% regression gate. No state-restoration mechanism is
implemented for this task.

## Results

**Stopped at quality: task correct counts are noninferior, but one new invalid
code output fails the no-new-invalid-output condition. No production adoption.**

| Task/check | Baseline | Candidate | Result |
| --- | ---: | ---: | --- |
| GSM8K correct | 36/64 | 36/64 | Noninferior; all requests unapplied |
| MMLU correct | 64/80 | 67/80 | Noninferior; 64 candidate requests applied |
| HumanEval functional | 138/164 | 138/164 | Noninferior; 1 candidate request applied |
| HumanEval syntax-valid | 158/164 | 157/164 | One new invalid output; stop |

`HumanEval/129` is the new syntax-invalid case. Both arms used the same
445-token prompt and stopped at the original 384-token output limit.
The baseline extracted source parses, but fails an assertion. The candidate
response starts an unclosed Python Markdown fence; the unchanged extraction
rule leaves that fence in the source, which raises `SyntaxError`.
The existing `score.py::_syntax_valid` detects the regression. This is the
task/plan's pre-existing no-new-invalid-output rule, not a newly invented
accuracy threshold or a greedy-hash gate. The generation manifest summarized
the score/error gate but did not spell out this syntax clause; the explicit
user instruction and main plan remain authoritative. No separate tool-call
fixture was found or claimed as evaluated.

The candidate applied exactly 16 calls to `cached256_q160` across the 16
target attention layers for this case. The next unapplied request correctly
reset to zero. Total candidate coverage is 65/308 requests and 2,240 calls;
243/308 requests are unapplied regression checks, not candidate-quality
coverage. No broad numerical cause is inferred from this one output.

All 616 generation requests completed successfully, emitting 23,087 tokens
per arm. Returned token log probabilities are finite, and no empty answers
occurred. Both arms have six HumanEval truncations. GSM8K/MMLU answer parsers
reported no invalid answers. Original and adapted scoring agree on all saved
answers. All input/output lengths, fresh run IDs and recorded target scopes
were checked. The no-stdin judge results are explicitly excluded.

| Item | Final state |
| --- | --- |
| Cold32K TTFT | Previous 2.1733x pass preserved |
| Strict Math numerical diagnostic | Previous failure preserved |
| 308-case quality | Not passed: new syntax-invalid output; score counts noninferior |
| 32K content retention | Not evaluated; quality stop |
| Natural-generation decode/MTP | Not evaluated as a decode benchmark; quality counters retained |
| Fixed-state decode regression | Not evaluated; no suitable full-state replay found |
| Prefix reuse | Not evaluated; quality stop |
| Sequential soak | Not evaluated; quality stop |

## Supplement: interpretation of HumanEval/129

This supplement reviews the saved responses from `9c4e86d5d64b`; it does not
rerun generation or grading, repair an answer, or change any gate, score,
artifact or stopping decision above. Evidence is the `HumanEval/129` record
in each `quality-validated-salt/{baseline,candidate}.jsonl` and the corrected
HumanEval judgments in the existing raw archive.

Both responses have the same 445-token input, 384 output tokens and
`finish_reason=length`. Neither contains a completed solution:

| Observation | Baseline | Candidate |
| --- | --- | --- |
| Implemented body | Finds the minimum value and its position, then comments | Only `n = len(grid)`, then comments |
| Path construction / return | Neither implemented; implicitly returns `None` | Neither implemented |
| Saved-source result | Parses, but fails an assertion | Unclosed Python Markdown fence causes `SyntaxError` |

The extractor in `benchmarks/r5_quality/score.py::_human_source` removes only
complete fenced blocks. The candidate's opening fence has no closing fence,
so it remains in the extracted source. Thus the syntax-valid count difference
reflects formatting and extraction under truncation, not a formerly correct
answer becoming incorrect. Removing the opening fence would still leave an
unfinished function, not a correct solution; no such repair was scored.

The fence does violate the prompt's executable-Python-only, no-Markdown
instruction, so the format deviation is real. However, this single case is
not strong evidence of broader practical quality degradation or an attention
computation bug. HumanEval functional scores remain 138/164 in both arms.
The reported quality stop is the conservative application of the
no-new-invalid-output condition, not proof that the backend lost a correct
answer or should be categorically rejected. The earlier headline requires
this context. Qualification remains unresolved; this supplement authorizes
neither further experiments nor production adoption.

## Resource outcome and shutdown

The watcher recorded 220 samples. Maximum sampled cgroup RAM was
17,179,840,512 bytes (just below 16 GiB, during startup), not an exact peak or
a minimum-RAM requirement. Response-time samples were approximately
14.64--14.82 GiB. Maximum sampled tmpfs usage was 2,211,737,600 bytes; existing
compiler caches account for the bulk, while logs/answers stay on disk.
OOM, OOM-kill and OOM-group-kill deltas are all zero. The final post-shutdown
sample is 9,775,550,464 bytes, largely retained caches rather than a live model.

GPU allocator process-lifetime peaks are 28,869,044,224 allocated and
29,213,327,360 reserved bytes in both arms. They include preceding requests
and are not independent per-arm peaks. This short quality run is not a soak
or a memory-leak qualification.

After the quality decision, the dedicated API received SIGTERM at 11:41:57
UTC. The engine and API exited and the watcher was stopped. The shutdown log
contains an `EngineDeadError` in the output handler after intentional engine
teardown at 11:41:59; it is retained, not hidden as a clean error-free log.
There were no failed inference responses in the 616-request suite. RAM remains
capped at 16 GiB with swap disabled. No follow-on model experiment was run.

## Artifacts, reproduction and remaining work

- [Machine-readable final result](artifacts/gfx1201_r5_quality_operations_20260914.json)
- [Complete raw archive](artifacts/gfx1201_r5_quality_operations_20260914.tar.gz)

Archive SHA256:
`0895ec2002e5cb46a08012341f7c0dde67a90e0b0de0653b1888eef1422af510`.
The 7 MiB archive contains the fixed suite and manifests, all 616 full
responses with original prompts/references, request counters and resources,
server logs including preparation/shutdown, both invalid no-stdin and corrected
HumanEval scores, positive/negative judge controls, scoring-equivalence checks,
and actual executed client sources. `paired-score-partial-73-mmlu.json` is an
explicitly partial intermediate; `paired-score-complete.json`,
`quality-report-final.json` and `final-audited.json` are the complete results.
The archived client differs from the repository copy only in formatting of
the HTTP-error handler. Original scoring helper hashes match `.tools` exactly;
the repository judge is the corrected stdin-open implementation.

Raw disk directory:
`/srv/ai/cache/radiance-qwen38-tp1/r5-quality-20260914`.
Do not extract the uncompressed 68 MiB archive into tmpfs for a larger run.
The compressed copy committed here does not include SDKs, model weights or
compiler caches. There are no profiler traces.

To reproduce generation, use the same isolated, pinned environment. Verify
the copied hook hash before loading one model with
`benchmarks/launch_gfx1201_r5_model_ab.sh`, and direct its log, PID and stats
arguments to a fresh disk-backed trial directory. Run `monitor.py` before
startup with that PID file; do not raise the memory cap if startup fails.
Then, inside the container, run the following with fresh paths and a unique
trial ID (the preserved suite must retain its exact SHA256):

```sh
/tmp/tq-venv/bin/python /cache/TRIAL/harness/run.py \
  --suite /cache/TRIAL/suite-308.jsonl \
  --output /cache/TRIAL/quality --run-id UNIQUE-TRIAL \
  --stats /cache/TRIAL/stats.json
```

On the host, use `.venv/bin/python` and the checked-in helpers to verify the
judge before grading. `--original` points to the extracted, unchanged helper:

```sh
.venv/bin/python benchmarks/r5_quality/judge_controls.py \
  --original /DISK/EXTRACTED/harness/judge_humaneval.py \
  --image sha256:3cead535c32c11c59b222bbaa502076b9c737fb969b70b7bc9bb8ce77da5698e \
  --output /DISK/TRIAL/judge-controls.json

.venv/bin/python benchmarks/r5_quality/judge_humaneval.py \
  --input /DISK/TRIAL/quality/baseline.jsonl \
  --output /DISK/TRIAL/quality/baseline-humaneval.json \
  --image sha256:3cead535c32c11c59b222bbaa502076b9c737fb969b70b7bc9bb8ce77da5698e
```

Grade the candidate file identically, without regenerating answers. Aggregate
with `report.py --suite ... --directory /DISK/TRIAL/quality --output ...`.
It requires all 308 IDs and all 164 isolated HumanEval judgments per arm,
and reports task-score and invalid-output gates separately. `finalize.py`
reproduces this run's explicit quality-stop summary using the archived layout.
No repeated grading result without the successful negative control is valid.

Validation: six focused CPU tests pass; pre-commit passes on the changed code
and documents. The real isolated positive/negative controls and all 328 real
HumanEval executions completed. Original/adapted scorer equivalence holds
for all 616 current answers, and input/output/count/scope invariants hold for
all 616 requests. New format failures are not repaired or filtered away.

The remaining adoption work is unresolved output-format qualification and
the unexecuted 32K retention, decode, prefix-reuse and operational gates.
Fixed-state decode additionally needs a suitable existing replay mechanism
or a separately approved plan. This task ends here: no precision tuning,
new backend, production registration, default enablement or merge follows.
