# 3DCodeBench runtime

All executable helpers are part of Gym. No external benchmark checkout or
workspace sibling is read. `render.py` and `export_glb.py` are pinned native
Blender helpers uploaded from Gym into each fresh prediction-verification sandbox.

- `setup_assets.py --output DIRECTORY` downloads the pinned data and prepares task JSONLs.
- `prepare_references.py --prepared-root DIRECTORY` exports and validates trusted meshes using the bundled exporter.
- `download_weights.py [--cache-dir DIRECTORY]` downloads learned checkpoints using the standard Hugging Face cache.
- `check_scorer_models.py` loads Gym's bundled Uni3D encoder and its checkpoint.
- `check_sandbox.py IMAGE OUTPUT` verifies a valid and invalid program through real Gym sandboxes.
- `install.sh`, `build-openvdb.sh`, and `install-scorer.sh` run inside disposable image builds.
- `validate_blender.py` checks OpenVDB, Cycles CPU rendering, numerical dependencies, and GLB export.

Dataset and reference locations are explicit arguments. Model weights use
`HF_HOME` if supplied, otherwise Hugging Face's normal cache. Build validation
reports use `THREED_VALIDATION_DIR` if supplied, otherwise `/var/log/3dcodebench`
inside the build container. There is no required relationship between these paths.

The FanCoral repair sizes sparse arrays from its frozen endpoint list in a derived
program. The shipped Elkhorn specialization exports zero faces; its published
canonical seed-zero geometry GLB from the same dataset revision is downloaded
with a checked SHA-256. Both repairs record provenance beside reference GLBs and
preserve the original reference programs. Agent inputs contain only descriptions
and allowed images; reference geometry and programs remain trusted verifier data.
