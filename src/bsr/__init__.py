from typing import Final, Optional

import sys
from importlib import metadata as importlib_metadata

import bpy

from ._camera import Camera
from ._light import Light

# Exposed functions and classes (API)
# Note: These should not be imported within the package to avoid circular imports
from .blender_commands.file import load, reload, save
from .blender_commands.macros import (
    clear_materials,
    clear_mesh_objects,
    deselect_all,
    scene_update,
)
from .frame import FrameManager
from .geometry.composite.pose import Pose
from .geometry.composite.rod import (
    AnnulusRodWithSpline,
    FinnedRodWithSpline,
    RectAnnulusRodWithSpline,
    Rod,
    RodWithBox,
    RodWithCylinder,
    RodWithSpline,
)
from .geometry.composite.stack import (
    AnnulusRodStack,
    RectAnnulusRodStack,
    RodStack,
    SplineFinnedRodStack,
    SplineRodStack,
    create_annulus_rod_collection,
    create_rect_annulus_rod_collection,
    create_rod_collection,
    create_spline_finned_rod_collection,
    create_spline_rod_collection,
)
from .geometry.primitives.pipe import (
    BezierSplineAnnulusPipe,
    BezierSplinePipe,
    BezierSplineRectAnnulusPipe,
)
from .geometry.primitives.simple import Cylinder, Plane, Sphere
from .viewport import find_area, set_view_distance


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: Final[str] = get_version()
frame_manager = FrameManager()
camera = Camera()
light = Light()
