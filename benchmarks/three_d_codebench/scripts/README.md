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
Default GPU jobs use four GPUs, TP=4, four concurrent episodes, 131072 context,
50 build steps, an eight-image request cap, and a four-hour generation allocation.
Mounts default to the explicit data, reference, clone, checkpoint, output, and
optional cache paths. `THREED_CONTAINER_MOUNTS` can supply extra mounts, including
symlink targets.
