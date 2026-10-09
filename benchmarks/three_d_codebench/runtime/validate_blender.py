# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate native ARM Blender, Cycles rendering, GLB export, and numerical libs."""

import json
import os
from pathlib import Path

import bpy
import numpy
import scipy
import shapely
import skimage


assert bpy.app.version[:2] == (5, 0), bpy.app.version_string
assert bpy.app.build_options.openvdb, "Blender must enable OpenVDB for voxel remeshing"
output = Path(os.environ.get("THREED_VALIDATION_DIR", "/var/log/3dcodebench"))
output.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
bpy.ops.mesh.primitive_uv_sphere_add()
bpy.context.object.data.remesh_voxel_size = 0.15
bpy.ops.object.voxel_remesh()
assert len(bpy.context.object.data.polygons) > 0
bpy.ops.export_scene.gltf(filepath=str(output / "sphere.glb"), export_format="GLB")
bpy.ops.object.camera_add(location=(3, -3, 2))
camera = bpy.context.object
camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
scene = bpy.context.scene
scene.camera = camera
bpy.ops.object.light_add(type="AREA", location=(2, -2, 4))
bpy.context.object.data.energy = 500
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 4
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.filepath = str(output / "sphere.png")
bpy.ops.render.render(write_still=True)
assert (output / "sphere.glb").stat().st_size > 0
assert (output / "sphere.png").stat().st_size > 0
(output / "runtime.json").write_text(
    json.dumps(
        {
            "blender": bpy.app.version_string,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "shapely": shapely.__version__,
            "skimage": skimage.__version__,
            "openvdb": True,
            "voxel_remesh": True,
            "cycles_cpu": True,
            "glb_export": True,
        },
        indent=2,
    )
    + "\n"
)
print("Blender ARM runtime validation passed")
