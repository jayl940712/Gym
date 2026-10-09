# 3DCodeBench in Gym

This integration is self-contained in Gym. The six metric implementations live
in `metrics/`, Blender rendering and mesh export live in `runtime/`, prompts live
in `prompts/`, and the Uni3D inference encoder lives in `metrics/uni3d/`.
An external 3DCodeBench or Uni3D checkout is never needed. Removing a sibling
`3dcodebench` directory does not affect generation, verification, or scoring.
Third-party licenses and pinned source provenance are in `third_party/`.

The benchmark has 212 text-to-3D and 212 image-to-3D tasks. Dataset assets and
learned weights are downloaded during setup; they are not executable source
repositories. Dataset, reference, model-cache, container, and result paths are
independent settings. No shared workspace tree or author's directory is required.

## Start from a fresh clone

```bash
git clone --branch 3dcodebench-opencode https://github.com/jayl940712/Gym.git
cd Gym
uv sync --extra dev --extra sandbox

# Choose any writable dataset directory. This default is inside the clone.
uv run python -m benchmarks.three_d_codebench.runtime.setup_assets \
  --output data/three_d_codebench
```

Setup downloads only the benchmark reference programs, original descriptions,
and published input images from `YipengGao/3DCode` revision
`0db485b63a2991231f0fad6c6e0dc01bcbbced1e`. It generates `benchmark.jsonl`,
`text_to_3d.jsonl`, `image_to_3d.jsonl`, and `smoke-5-each.jsonl`, with the
associated task folders and private `reference/`. The `.downloads` directory is
an internal download cache, not a required user-managed workspace layout.

## Runtime and reference geometry

Use Docker, or another Gym sandbox provider configured with a compatible runtime.
Build the supplied Blender/OpenCode image from the Gym repository root:

```bash
docker build -f benchmarks/three_d_codebench/Dockerfile.runtime \
  -t 3dcodebench-runtime:latest .

# Generate trusted reference meshes with the Gym-owned exporter.
docker run --rm \
  -v "$PWD:/gym" -v "$PWD/data/three_d_codebench:/dataset" \
  3dcodebench-runtime:latest python3 \
  /gym/benchmarks/three_d_codebench/runtime/prepare_references.py \
  --prepared-root /dataset --limit 0
```

The runtime installs OpenCode 1.17.11 and Blender 5.0.1 with OpenVDB and CPU Cycles.
The build recipe supports Debian Linux; ARM64 may require rebuilding Blender
with OpenVDB. A compatible existing image can be selected with `THREED_TASK_IMAGE`
instead. Reference preparation validates surface meshes, retains valid cached
exports, and records the FanCoral and Elkhorn compatibility provenance. Original
reference programs remain unchanged and never enter agent sandboxes.

## Configure the policy and run Gym

Start your chosen policy endpoint, then create a Gym config in the clone:

```bash
cat > 3dcodebench-local.yaml <<'YAML'
config_paths:
- benchmarks/three_d_codebench/config.yaml
- responses_api_models/vllm_model/configs/vllm_model.yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: EMPTY
policy_model_name: your-served-model-name
YAML

# Override these if your assets are elsewhere. Paths can be independent.
export THREED_DATASET="$PWD/data/three_d_codebench"
export THREED_TASK_IMAGE=3dcodebench-runtime:latest
uv run gym env start --config 3dcodebench-local.yaml
```

In a second shell with the same config and environment:

```bash
uv run gym eval run --no-serve --resume --config 3dcodebench-local.yaml \
  --agent three_d_codebench_agent \
  --input data/three_d_codebench/smoke-5-each.jsonl \
  --output results/3dcodebench-smoke/rollouts_generation.jsonl \
  --num-repeats 1 --concurrency 4
```

Use `benchmark.jsonl` for all 424 tasks, `text_to_3d.jsonl` for text only, or
`image_to_3d.jsonl` for image only. Generated code is independently executed and
rendered in a fresh sandbox. Successful generation produces provisional execution
rewards; the next step computes the final learned metrics.

## Score with Gym-owned metrics

Install scoring dependencies into a Python environment compatible with your CUDA
and PyTorch installation. The inference encoder is included in Gym; no git clone
of another code repository occurs.

```bash
uv pip install -r benchmarks/three_d_codebench/requirements-scoring.txt
# Authenticate an approved account for the gated DINOv3 weights if needed.
uv run python -m benchmarks.three_d_codebench.runtime.download_weights

uv run python -m benchmarks.three_d_codebench.score \
  --run results/3dcodebench-smoke \
  --references data/three_d_codebench/reference --task all --device cuda

uv run gym eval profile \
  --inputs results/3dcodebench-smoke/scoring_inputs.jsonl \
  --rollouts results/3dcodebench-smoke/rollouts.jsonl
```

Hugging Face's normal cache is used; `HF_HOME` or the downloader's `--cache-dir`
can select any cache directory. If downloading with `--cache-dir`, set `HF_HOME`
to that same directory when scoring. `Dockerfile.scorer` provides an alternative image
build using a CUDA/PyTorch base selected with `--build-arg BASE_IMAGE=...`.
Data and reference paths do not have to be adjacent to the weights or images.

The scorer supports `--task text_to_3d`, `--task image_to_3d`, and `--task all`
(default). It requires exactly one completed rollout for every selected input and
preserves repeats and model failures. There is no `--source` option: all scorers
resolve from this installed Gym package.

| Metric | Text-to-3D | Image-to-3D |
| --- | --- | --- |
| SigLIP2 / final Gym reward | Original description versus generated renders; mean across four views | View-paired generated versus reference images |
| Uni3D conditioning | Description versus generated point cloud | Canonical reference image versus generated point cloud |
| Uni3D shape | Reference versus generated point cloud | Same |
| Chamfer | Reference/generated geometry, direct and yaw-minimum distances | Same |
| Executability / failure taxonomy | Blender render status and error breakdown | Same |
| DINOv3 | Supplementary reference-image diagnostic | Generated versus reference images |

Text scoring reads the unmodified dataset description, not the expanded OpenCode
workflow prompt. Conditional and penalized native aggregates are preserved;
there is no upstream combined reward. Learned similarities are not accuracy
percentages, and text/image rewards should be compared within their own tracks.

## Resume and artifacts

Rerun the same `gym eval run --resume` command to retain completed episodes and
restart interrupted episodes. Keep inputs, repeat count, config, and output path
consistent. Scoring reuses reports only when inputs, scorer code, and options
match its cache fingerprint; Uni3D's presence-only cache is bypassed on recompute.

Each result directory contains `rollouts_generation.jsonl`, final `rollouts.jsonl`,
`native_metrics.json`, `evaluation_summary.json`, `artifacts/`, and `opencode/`.
OpenCode exports save image-read attachments; evaluator renders in `artifacts/`
are produced independently and are not automatically returned to the actor.

## Optional Slurm / Pyxis execution

Reusable [Slurm scripts](scripts/README.md) are included for clusters. They use
explicit dataset, reference, runtime-image, scorer-image, serving-image, and
checkpoint settings. They do not require the standalone Docker flow, an author's
containers, or a particular state-directory layout. Account, partition, QoS,
mounts, and policy settings are supplied for your cluster.

See [agent documentation](../../responses_api_agents/three_d_codebench_agent/README.md)
and [runtime documentation](runtime/README.md) for implementation details.
