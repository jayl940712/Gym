# Hosted Responses model

A hosted OpenAI Responses endpoint for Gym clients, including OpenCode's chat-compatible
provider. `/v1/responses` passes through the existing authenticated `openai_model` client.
`/v1/chat/completions` translates requests to Responses and converts the buffered result
back to chat. All upstream requests use `/v1/responses`; the host need not offer chat.

Responses-only message phase metadata is omitted when emitting chat replies.
The adapter preserves images, function-call IDs/results, visible reasoning summaries,
text, token usage, and output-limit finish reasons. It uses Gym's aiohttp client and
existing retry/concurrency/timeout configuration. It does not stream upstream tokens;
Gym streams the complete converted result to the harness.

Use `configs/hosted_responses_model.yaml` with `policy_base_url`, `policy_api_key`, and
`policy_model_name`. Keep credentials in environment interpolation. OpenCode should use
`@ai-sdk/openai-compatible`, with model `interleaved.field: reasoning_content` if the
host returns reasoning summaries.

The CPU 3DCodeBench launcher demonstrates this configuration:
`benchmarks/three_d_codebench/scripts/run-3dcodebench-hosted-cpu.slurm`.
