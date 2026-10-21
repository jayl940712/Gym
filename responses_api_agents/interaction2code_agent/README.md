# Interaction2Code agent

Standalone interactive webpage reconstruction with Gym's sandboxed OpenCode
harness and native Interaction2Code visual metrics.

See the [evaluation guide](../../fern/versions/latest/pages/evaluation-tutorials/interaction2code.mdx)
and [benchmark configuration](../../benchmarks/interaction2code/config.yaml).

The agent receives only `before.png`, `after.png`, and `placeholder.jpg`. It derives
screenshot dimensions directly from the input images. No evaluator or screenshot helper code is
uploaded to the coding sandbox; the agent writes its own browser automation. The task prompt asks the model to render and read both
states, revise `index.html`, and repeat. For small differences, it also asks the agent
to read matching reference/generated close-up crops. Preview files under `/workspace/feedback/`
are retained before the coding sandbox is destroyed. They are for trajectory review;
the verifier independently renders the final HTML in a separate sandbox. Supplied
assets must remain unchanged. An external `sha256sum` check records
`input_assets_audit.json` before cleanup; its result is not sent to the policy and
does not change the native score.

See [hosted CPU smoke tests](../../benchmarks/interaction2code/scripts/README.md#hosted-kimi-cpu-smoke-tests)
for Kimi K3 inference, reasoning replay, and full model-call capture. Re-export prepared
JSONL inputs to use the revised prompt; changing the source prompt does not rewrite
previously prepared datasets or old results.

## Persistent native scoring

The trusted verifier uses a bounded pool of persistent CPU scoring workers.
Each worker lazily loads CLIP ViT-B/32 and English EasyOCR once, then reuses
both models across tasks. `scoring_concurrency` sets the worker count per agent-server
process; `scoring_threads` controls PyTorch threads in each worker. No new
container is needed: workers use the existing isolated Python scoring runtime.
The first request in each worker includes model initialization; later requests
reuse its models. These workers receive only trusted dataset/artifact paths,
never policy requests or generated HTML execution.

Each worker handles one task at a time. Timeout includes waiting for a free
worker, initialization, and scoring. Timed-out, cancelled, or crashed workers
are stopped; their next request starts a replacement. Metric errors are returned
without discarding an otherwise healthy worker. Workers stop when the Gym agent
server shuts down. Per-task diagnostics remain in `scorer.stderr.txt`; startup
and worker crash logs are under `<runtime_root>/worker-logs/`.

The standalone `scorer.py` CLI remains available for independent parity checks.
Metric definitions, rewards, and browser sandbox isolation are unchanged.

Reference OCR text is cached by image content in
`<runtime_root>/models/ocr-text/`. Cache entries include the exact OCR string,
including whitespace and empty strings. Changes to scorer source, package versions,
CPU architecture, configured threads, or model weights invalidate the cache. Shared
workers coordinate writes with per-image locks and atomic publication. Corrupt or
unavailable cache entries are recomputed. Generated images always receive fresh OCR.
Set `I2C_OCR_CACHE=0` to disable reuse for independent parity checks. OCR stays on CPU.

## Generation modes

`agent_mode` defaults to `opencode`, which retains the existing tool-using,
visual-feedback workflow. Set `agent_mode: single_turn` for one system prompt
and one user turn containing the unannotated before/after screenshots. There is
no starting HTML in this dataset. The single-turn prompt asks for a complete,
self-contained HTML document in one fenced `html` block, using the same relative
placeholder image and interaction-trigger convention as OpenCode mode. No tools,
reference annotations, evaluator instructions, or helper scripts are supplied.

The single-turn response is extracted from assistant text, excluding reasoning.
A missing/ambiguous fence or incomplete HTML is a task failure; generation cut off
by the output limit is recorded as `context_limit`. There is no repair conversation.
The extracted `index.html` uses the same isolated browser verifier and native
metrics as OpenCode. Only verification creates a sandbox in single-turn mode.

`single_turn_max_output_tokens` defaults to 32768. Task artifacts include the actual
`input.json`, `model_request.json`, raw `model_response.json`, normalized
`response.json`, assistant text, extracted HTML, native renders, and scores.
The standard model-call capture also records the one logical model request/response.
No OpenCode session export or visual-feedback trajectory is expected in this mode.

The historical `protocol` identifier is unchanged for compatibility with existing
prepared data. Use `agent_mode` in results to distinguish generation modes.
