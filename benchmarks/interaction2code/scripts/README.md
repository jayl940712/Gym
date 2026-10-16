# Slurm evaluation

These launchers use Enroot/Pyxis instead of Docker. Build the browser image from
Gym alone; the job checks Firefox clicks, Chromium, and OpenCode before publishing it.
Playwright launches Debian Chromium through a wrapper so the default
`chromium.launch()` works on ARM hosts with 64 KiB kernel pages:

```bash
sbatch -A YOUR_ACCOUNT benchmarks/interaction2code/scripts/build-browser.slurm "$PWD" /path/interaction2code-opencode.sqsh
```

Run from the Gym checkout root (the scripts themselves are under
`benchmarks/interaction2code/scripts`). Prepare assets and the CPU scorer using
the benchmark evaluation guide, then set the following variables:

```bash
export I2C_DATASET=/path/to/prepared-dataset
export I2C_SCORER_ROOT="$PWD/benchmarks/interaction2code/.cache/scorer"
export I2C_TASK_IMAGE=/path/interaction2code-opencode.sqsh
export I2C_SERVING_IMAGE=/path/vllm.sqsh
export I2C_CHECKPOINT=/path/policy-checkpoint
export I2C_SERVING_PYTHON=/path/to/python/inside/serving-image
export I2C_TOOL_CALL_PARSER=qwen3_coder
export I2C_REASONING_PARSER=deepseek_r1
export I2C_HF_OVERRIDES='{"architectures":["NemotronH_Super_Omni_Reasoning_V3"]}'
# Include any mounts needed to resolve checkpoint symlinks and shared assets.
export I2C_CONTAINER_MOUNTS=/lustre:/lustre,/scratch:/scratch
sbatch -A YOUR_ACCOUNT benchmarks/interaction2code/scripts/run-interaction2code.slurm \
  "$PWD" /path/results "$I2C_DATASET/interaction2code_benchmark.jsonl" 0 1
```

The GPU job uses partition `batch`, QoS `interactive`, TP=4, 131072 context,
64 coding episodes, 64 vLLM sequences, 8192 batched tokens, 0.85 GPU memory
utilization, and CUDA graph captures `[1,2,4,8,16,32,48,64]`. It runs eight CPU
scoring processes with two threads each. Up to 32 images per model request allow
reference pairs and repeated visual feedback to remain in conversation history. Browser verification runs in a fresh
sandbox with no reference annotations. The serving model is released when the
job ends; all native metrics are calculated during rollout collection.

For `qwen3_coder`, the launcher loads Gym's parser plugin for vLLM 0.24.
It skips incremental JSON argument conversion when parsing a complete,
non-streaming response and scans parameter tags without repeated searches.
This prevents large HTML tool calls from blocking the API event loop. Normal
tool arguments and native metric definitions are preserved. Other tool parser
names do not load this plugin.

Use a limit of `5` instead of `0` for a smoke evaluation in a separate output
folder. Reuse the same output folder and exact inputs to resume; completed rows
are preserved. Failed task results also count as completed. To retry a failure,
remove its row and associated episode/export artifacts first. Review masked
infrastructure failures: the final coverage report rejects them and incomplete
runs. Results include `rollouts_generation.jsonl`, per-episode artifacts,
OpenCode trajectories, logs, `summary.json`, and a `complete` marker.

The visual-feedback prompt asks the agent to write its own browser automation and
inspect both generated states. No evaluator or capture helper is uploaded to its
coding sandbox. Enroot shares host networking. If an agent starts its own server, it should use port `0` and stop only its
recorded PID. Validate concurrent server isolation with:

```bash
python benchmarks/interaction2code/runtime/check_sandbox.py /path/browser.sqsh /path/check-results
```

## Hosted Kimi CPU smoke tests

The hosted launcher requires no serving image, checkpoint, or local GPU. It reuses the
browser/OpenCode image and existing CPU scorer runtime. Set the dataset, task image,
and scorer variables above, then authenticate through `WEBARENA_JUDGE_API_KEY` in the
job environment (or your `.bashrc`). The key is resolved from the environment and is
not written to the saved configuration.

```bash
export I2C_POLICY_MODEL=nvidia/moonshotai/kimi-k3
export I2C_POLICY_BASE_URL=https://inference-api.nvidia.com/v1
export I2C_CAPTURE_MODEL_CALLS=1
export I2C_OPENCODE_STEPS=30
sbatch -A YOUR_ACCOUNT --cpus-per-task=16 --mem=24G --time=01:30:00 \
  benchmarks/interaction2code/scripts/run-interaction2code-hosted-cpu.slurm \
  "$PWD" /path/kimi-test /path/ten-sampled-inputs.jsonl 10 1
```

The default partition is `cpu`, QoS `cpu-interactive`, with 32 CPUs, 128 GiB RAM,
and zero GPUs. Model/agent concurrency is 64; ten selected tasks produce at most ten
simultaneous episodes. Context compaction and pruning are disabled, and OpenCode
replays exposed reasoning using `reasoning_content`. Each call has a 32768-token
output limit and a 131072-token context window. Set `I2C_OPENCODE_STEPS=30` for a
shorter prompt experiment; the default is 50 model steps. Provider-internal reasoning handling
is outside Gym's control.

The launcher also opts into bounded upstream retries: a completely empty chat reply
with reported zero tokens is treated as a provider error and retried, with at most
three attempts. Valid text, reasoning, tool calls, and refusals are preserved. Other
Gym model configurations keep the previous behavior by default. These retries happen
inside a logical Gym model call; boundary captures are not a per-attempt upstream wire log.

Full request/response bodies, including inline images, are saved under
`model_calls/*.capture.jsonl`. These files may be large because each call includes
conversation history. Native OpenCode session exports are under `opencode/`.
Per-episode `artifacts/episodes/*/feedback/` preserves generated preview rounds.
The independent evaluator still captures the final page in a fresh sandbox and runs
the unchanged native Interaction2Code scorer. Use a new output directory when changing
prompts. Reuse identical inputs and configuration to resume unfinished cases.

Audit actual successful screenshot reads and page revisions after a run:

```bash
python -m benchmarks.interaction2code.audit_visual_feedback --run /path/kimi-test
```

The audit distinguishes generated-image inspection (`visual_feedback_used`), a later
HTML mutation (`post_feedback_html_revision`), and at least two inspected rounds with
an intervening revision (`visual_feedback_verified`). A task may score well while
stopping after one preview; the audit records that behavior rather than treating the
model's final claim as evidence of another round.
