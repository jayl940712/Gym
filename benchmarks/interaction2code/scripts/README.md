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

Enroot isolates process namespaces but shares host networking. The agent prompt
starts its HTTP server on port `0`, reads the assigned URL from its log, and
stops only its recorded server PID. Fixed ports can collide across episodes.
The Slurm launcher allows 128 Enroot commands so 64 long coding commands can
run alongside sandbox setup and verification. Validate concurrent servers with:

```bash
python benchmarks/interaction2code/runtime/check_sandbox.py /path/browser.sqsh /path/check-results
```
