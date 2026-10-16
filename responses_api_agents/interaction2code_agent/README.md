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
