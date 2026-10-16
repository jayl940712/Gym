# Optional 3DCodeBench Slurm scripts

Run from your own Gym clone. Slurm/Pyxis/Enroot and compatible compute nodes are
required. Override `sbatch -A ACCOUNT -p PARTITION -q QOS` for your cluster.
No script uses `THREED_WORKSPACE` or expects a sibling 3DCodeBench checkout.

| Script | Positional arguments |
| --- | --- |
| `build-3dcodebench.slurm` | Output runtime `.sqsh` path; optional `THREED_RUNTIME_BASE` |
| `build-3dcodebench-scorer.slurm` | Output scorer `.sqsh` path; requires `THREED_SCORER_BASE` |
| `build-3dcodebench-openvdb.slurm` | Existing image, new output image |
| `prepare-3dcodebench-references.slurm` | Prepared dataset directory, optional object limit (default 0 = all) |
| `check-3dcodebench.slurm` | Runtime image, output report directory |
| `run-3dcodebench.slurm` | Gym clone, output directory, input JSONL, limit, repeats, optional `--task` |
| `score-3dcodebench.slurm` | Reference directory, existing rollout directory, optional `--task` |

Runtime/scorer images can be anywhere. Set `THREED_TASK_IMAGE` and
`THREED_SCORER_IMAGE`; set `THREED_DATASET` to the prepared data directory and
optionally `THREED_REFERENCES` to independently located reference assets.
The dataset default is `data/three_d_codebench` inside Gym. Set
`THREED_SERVING_IMAGE`, `THREED_CHECKPOINT`, and, when needed,
`THREED_SERVING_PYTHON` and `THREED_HF_OVERRIDES` for your policy runtime.
Set `THREED_TOOL_CALL_PARSER` for the served policy and optionally
`THREED_REASONING_PARSER`; the launcher does not assume a model family.
`HF_HOME` optionally selects the learned-checkpoint cache.

```bash
export THREED_GYM_REPO="$PWD"
export THREED_DATASET=/your/dataset
export THREED_TASK_IMAGE=/your/runtime.sqsh
export THREED_SCORER_IMAGE=/your/scorer.sqsh
export THREED_SERVING_IMAGE=/your/vllm.sqsh
export THREED_CHECKPOINT=/your/policy-checkpoint
export THREED_TOOL_CALL_PARSER=your_vllm_tool_parser
scripts=benchmarks/three_d_codebench/scripts
sbatch -A YOUR_ACCOUNT "$scripts/run-3dcodebench.slurm" "$PWD" \
  "$PWD/results/3dcodebench" "$THREED_DATASET/benchmark.jsonl" 0 1 --task all
sbatch -A YOUR_ACCOUNT "$scripts/score-3dcodebench.slurm" \
  "$THREED_DATASET/reference" "$PWD/results/3dcodebench" --task all
```

The second command is a scoring-only alternative, submitted after generation
completes. Do not submit both concurrently to one output directory.
Use `THREED_GENERATION_ONLY=1` to separate generation and scoring allocations.
Task selection is `all` by default; text/image selections filter before the limit.
Reuse the exact generation command/output to resume partial work. Scripts do not
resubmit themselves. Completed runs are no-ops; concurrent writers are rejected.
Default GPU jobs use four GPUs, TP=4, 16 concurrent episodes, 131072 context,
50 build steps, an eight-image request cap, and a four-hour generation allocation.
Mounts default to the explicit data, reference, clone, checkpoint, output, and
optional cache paths. `THREED_CONTAINER_MOUNTS` can supply extra mounts, including
symlink targets.

## Hosted Responses policy on CPU

`run-3dcodebench-hosted-cpu.slurm` runs OpenCode and Blender on the `cpu`
partition with `cpu-interactive` QoS, 32 CPUs, 128 GB RAM, and no GPUs. It reuses
`THREED_TASK_IMAGE` and `THREED_SCORER_IMAGE` and scores with `--device cpu`.
Learned scoring is slower on CPU. No vLLM image or local policy checkpoint is needed.

Export `WEBARENA_JUDGE_API_KEY` before submission, or define it in your `.bashrc`.
The script references the credential through an environment interpolation; it does
not write the credential to its saved config. `THREED_POLICY_BASE_URL` defaults to
`https://inference-api.nvidia.com/v1`, and `THREED_POLICY_MODEL` defaults to
`azure/openai/gpt-6.1-sol`. OpenCode uses its chat-compatible provider through Gym's `hosted_responses_model`
adapter; the upstream call still uses `/v1/responses`. It uses a 131072-token
context window, 32768 output tokens per call, and at most 50 build steps.
Concurrency remains 64; a ten-task selection launches at most ten episodes.

```bash
sbatch -A YOUR_ACCOUNT benchmarks/three_d_codebench/scripts/run-3dcodebench-hosted-cpu.slurm \
  "$PWD" "$PWD/results/hosted-image10" "$THREED_DATASET/benchmark.jsonl" \
  10 1 --task image_to_3d
```

This launcher defaults to ten image-to-3D tasks. Explicit `--task` and limit
arguments select other subsets. Reuse the same output and arguments to resume.
Use `THREED_GENERATION_ONLY=1` if scoring should run in a separate allocation.


For a hosted chat-completions model, set `THREED_POLICY_API=chat`. This selects
Gym's `openai_model` proxy and calls `/v1/chat/completions` directly instead of
translating requests to Responses. For example:

```bash
export THREED_POLICY_API=chat
export THREED_POLICY_MODEL=nvidia/moonshotai/kimi-k3
```

The image-to-3D prompt now uses two aligned render-and-inspect rounds with the
uploaded evaluator rendering helper. New preparation uses this prompt; existing
prepared JSONLs retain their saved prompts. To test the new prompt with existing
assets, replace only the image task's `responses_create_params.input` content with
`prompts/image_to_3d_system_prompt.txt` and save a new input JSONL. Keep older run
inputs intact. Preview images are collected under each artifact's `feedback/`;
final scored renders are independently regenerated under `renders/`.


For per-call multimodal trace capture, set `THREED_CAPTURE_MODEL_CALLS=1` before
submitting a new run. The launcher enables Gym observability and stores full
Gym-boundary model request/response bodies under `model_calls/*.capture.jsonl`,
including inline image payloads and tool schemas. It explicitly disables OpenCode
compaction and pruning and keeps reasoning replay configured. Capture mode is part
of the resume settings; reuse the same mode when resuming. OpenCode exports and
preview artifacts are retained alongside the raw call captures. Hosted provider
internal prompt rendering and token IDs are outside this capture boundary.
