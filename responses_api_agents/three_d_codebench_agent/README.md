# 3DCodeBench OpenCode agent

This agent runs text-to-3D and image-to-3D reconstruction through Gym's
[sandboxed OpenCode harness](../opencode_sandboxed_agent/README.md).
See the [benchmark README](../../benchmarks/three_d_codebench/README.md) for
standalone setup and the rollout/scoring commands. The wiring is in
[config.yaml](../../benchmarks/three_d_codebench/config.yaml).

## Input format

Each prepared JSONL row identifies a task and contains Responses API input:

```json
{
  "task": "text_to_3d",
  "record_id": "AgaveMonocot_seed0",
  "responses_create_params": {
    "input": [{"role": "user", "content": "Prepared benchmark prompt..."}]
  }
}
```

`task` is `text_to_3d` or `image_to_3d`. The agent reads
`<dataset_root>/<task>/<record_id>/task.json` to validate the task and find its
conditioning images. Text tasks use the published object description. Image
tasks upload `Image_005.png`, `Image_015.png`, `Image_025.png`, and
`Image_035.png` to `/workspace/`; the prompt instructs OpenCode to inspect them
with its read tool. The agent does not upload reference programs or meshes.

## Execution and verification

1. Start a fresh sandbox from the configured runtime image and upload input PNGs.
2. Run OpenCode against the configured Gym model server. The model uses tools to
   write and test `/workspace/solution.py`.
3. Save the response, final assistant text, and generated file. The submitted
   program is `solution.py`; `answer.txt` records the response for inspection.
4. Destroy the generation sandbox. Upload only `solution.py` to a second fresh
   sandbox and run the trusted Blender renderer and GLB exporter bundled in Gym.
5. Download prediction images, geometry, and execution logs; destroy that sandbox.

The generation-stage reward is provisional executability. Offline
[score.py](../../benchmarks/three_d_codebench/score.py) replaces it with native
task-specific SigLIP2 similarity (text-to-render for text tasks, render-to-reference
for image tasks) and attaches the other benchmark diagnostics. There is no
dedicated `resources_servers/three_d_codebench` component: verification is
implemented by this agent's `_grade()` method and the offline scorer.

Missing programs, invalid Blender programs, and exhausted agent budgets remain
measured failures. Infrastructure errors are marked `mask_sample: true`.
The full launcher rejects incomplete runs before final scoring.

## Configuration and artifacts

Important settings are `model_server`, `dataset_root`, `results_dir`,
`artifacts_dir`, `sandbox_provider`, and `sandbox_config.image`. The Slurm launcher
sets these paths for the configured dataset/output directories and uses Enroot.
The checked-in configuration also supplies OpenCode settings: 50 build steps,
131072 context tokens, disabled automatic compaction, and a 300-second timeout
per independent render/export command. The policy checkpoint belongs to the
model server; learned metric weights belong to the offline scorer.

Each episode's `artifact_dir` holds `input.json`, `response.json`, `answer.txt`,
`solution.py` when available, render/export stdout and stderr, `renders/`,
`glb/`, and `result.json`. The result records status, provisional reward,
OpenCode execution details, and trajectory metadata.

Run the integration checks from the Gym root:

```bash
.venv/bin/python -m pytest benchmarks/three_d_codebench/test_integration.py
```

Real sandbox validation is provided by
[`check-3dcodebench.slurm`](../../benchmarks/three_d_codebench/scripts/check-3dcodebench.slurm).

All rendering/export helpers are uploaded from `benchmarks/three_d_codebench/runtime/`
into the fresh verification sandbox. Scorers are in `benchmarks/three_d_codebench/metrics/`;
no sibling repository or container-embedded benchmark source is required. Dataset
and sandbox image paths are configured independently.
