"""
Merge simulation output into a scene template.

Workflow:
1. Run construct_blender_file() to produce a simulation .blend.
2. Open scene.blend.
3. Append every Object and Collection from the simulation .blend.
4. Save the merged result under a new name.
"""

import os
from pathlib import Path

import bpy
import numpy as np

import bsr
from src.bsr.blender_commands.file import load, save
from src.elastica_blender.converter.npz2blend import construct_blender_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _srgb_to_linear(c: float) -> float:
    """Apply the IEC 61966-2-1 sRGB -> linear conversion to a single channel."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def hex_to_rgba(
    hex_color: str, alpha: float = 1.0
) -> tuple[float, float, float, float]:
    """Convert a sRGB hex color string (e.g. '#3399FF') to a linear RGBA tuple.

    Blender stores colors in linear space, so sRGB hex values must be converted.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b), alpha)


# ---------------------------------------------------------------------------
# Paths and appearance — edit these
# ---------------------------------------------------------------------------
ROD_COLOR_HEX = "#2C79D3"

# CAMERA_LOCATION = np.array([2.25, 0.01, 4.0])
CAMERA_LOCATION = np.array([3, -2, 1.5])
CAMERA_LOOK_AT = np.array([0.5, 0.0, 0.0])
CAMERA_ROTATION = 0  # degrees; set to None to skip rotation
CAMERA_FOCAL_LENGTH = 85.0  # mm — 24 wide, 50 standard, 85 portrait

RENDER = False  # set to False to skip rendering
RENDER_FOLDER = os.path.join(
    os.getcwd(), "renders"
)  # output folder (created if missing)
RENDER_FILENAME = (
    "frame.png"  # base filename; frame number appended automatically
)
RENDER_WIDTH = 3840  # pixels
RENDER_HEIGHT = 2160  # pixels

NPZ_PATH = "multiple_fin_pipe_thickness_1.5mm_length_110.0mm_flow_rate_0_fin_taper_1.0_sim_results_render.npz"  # simulation output
SIMULATION_BLEND = (
    os.path.splitext(NPZ_PATH)[0] + ".blend"
)  # intermediate output
SCENE_BLEND = "scene_with_support.blend"  # your template scene
OUTPUT_BLEND = "final_" + SIMULATION_BLEND  # merged output

# ---------------------------------------------------------------------------
# Step 1 — build the simulation .blend (no camera settings here)
# ---------------------------------------------------------------------------
construct_blender_file(
    NPZ_PATH,
    SIMULATION_BLEND,
    tags=None,
    fps=-1,
    # annulus_rod=True,
    # finned_spline_rod=True,
    # rect_annulus_rod=True,
    fin_pipe_bundle=True,
)

# ---------------------------------------------------------------------------
# Step 2 — open the scene template
# ---------------------------------------------------------------------------
load(SCENE_BLEND)

# ---------------------------------------------------------------------------
# Step 3 — append rod geometry from the simulation .blend (skip lights/cameras)
# ---------------------------------------------------------------------------
sim_path = Path(SIMULATION_BLEND).resolve()

with bpy.data.libraries.load(str(sim_path), link=False) as (src, dst):
    dst.objects = list(src.objects)

for obj in dst.objects:
    if obj is None:
        continue
    if obj.type in "LIGHT":
        bpy.data.objects.remove(obj, do_unlink=True)
        continue
    if obj.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(obj)

# ---------------------------------------------------------------------------
# Step 3b — apply uniform color to all rod objects
# ---------------------------------------------------------------------------
rgba = hex_to_rgba(ROD_COLOR_HEX)

for obj in bpy.context.scene.collection.objects:
    if obj.type not in ("CURVE", "MESH"):
        continue
    for mat in obj.data.materials:
        if mat is None:
            continue
        mat.diffuse_color = rgba
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = rgba
            bsdf.inputs["Roughness"].default_value = 0.8
            bsdf.inputs["Metallic"].default_value = 0.0
            bsdf.inputs["Alpha"].default_value = rgba[3]

# ---------------------------------------------------------------------------
# Step 3c — set camera
# ---------------------------------------------------------------------------
bpy.context.scene.camera = bsr.camera._camera  # set as active render camera
bsr.camera.location = CAMERA_LOCATION
bsr.camera.look_at = CAMERA_LOOK_AT
bsr.camera.focal_length = CAMERA_FOCAL_LENGTH
if CAMERA_ROTATION is not None:
    bsr.camera.rotate(CAMERA_ROTATION)

# ---------------------------------------------------------------------------
# Step 4 — save merged result
# ---------------------------------------------------------------------------
save(OUTPUT_BLEND)
print(f"Saved merged scene to: {OUTPUT_BLEND}")

# ---------------------------------------------------------------------------
# Step 5 — render all frames (optional)
# ---------------------------------------------------------------------------
if RENDER:
    bsr.camera.set_resolution(RENDER_WIDTH, RENDER_HEIGHT)
    bsr.camera.set_file_path(RENDER_FILENAME, folder_path=RENDER_FOLDER)

    frames = list(
        range(
            bpy.context.scene.frame_start,
            bpy.context.scene.frame_end + 1,
        )
    )
    print(f"Rendering {len(frames)} frames to '{RENDER_FOLDER}/'...")
    bsr.camera.render(frames=frames)
    print("Rendering complete.")
