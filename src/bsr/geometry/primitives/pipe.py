__doc__ = """
BezierSplinePipe: 3D pipe creation using Bezier splines with configurable handle types.

This module provides a robust implementation for creating smooth pipe-like objects
in Blender, with specific attention to preventing abnormal tip appearance that can
occur with improper handle type configurations.
"""
__all__ = [
    "BezierSplinePipe",
    "BezierSplineFinnedPipe",
    "BezierSplineAnnulusPipe",
]

from typing import TYPE_CHECKING, Any, cast

import warnings
from numbers import Number

import bpy
import numpy as np
from numpy.typing import NDArray

try:
    from scipy.interpolate import interp1d
except ModuleNotFoundError:  # pragma: no cover
    interp1d = None

from bsr.geometry.protocol import BlenderMeshInterfaceProtocol, SplineDataType
from bsr.tools.keyframe_mixin import KeyFrameControlMixin

from .utils import _validate_position, _validate_radii


class BezierSplinePipe(KeyFrameControlMixin):
    """
    Creates a 3D pipe using Bezier spline curves with proper bevel geometry.

    This class creates smooth pipe-like objects by using Bezier splines with
    configurable handle types to avoid abnormal tip appearance common with
    VECTOR handles.

    Parameters
    ----------
    positions : NDArray
        The position of the spline control points. Shape: (3, n)
    radii : NDArray
        The radius at each control point. Shape: (n-1,)
    """

    input_states = {"positions", "radii"}
    name = "bspline"

    def __init__(
        self,
        positions: NDArray,
        radii: NDArray,
        downsample_num_element: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Spline constructor

        Parameters
        ----------
        positions : NDArray
            The position of the spline object. (3, n)
        radii : NDArray
            The radius of the spline object. (n,)
        downsample_num_element : int | None, optional
            If provided, downsample the positions and radii to this number of elements.
            Default is None (no downsampling).
        """
        assert (
            downsample_num_element is None or downsample_num_element >= 2
        ), "downsample_num_element must be at least 2 to include both endpoints"
        self.downsample_num_element = downsample_num_element
        number_of_points = positions.shape[1]
        if downsample_num_element is not None:
            number_of_points = min(number_of_points, downsample_num_element)
        self._obj = self._create_bezier_spline(number_of_points)
        self._obj.name = self.name
        self.update_states(positions, radii)

        self._material = bpy.data.materials.new(
            name=f"{self._obj.name}_material"
        )
        self._obj.data.materials.append(self._material)

    @classmethod
    def create(
        cls,
        states: SplineDataType,
        downsample_num_element: int | None = None,
    ) -> "BezierSplinePipe":
        """
        Basic factory method to create a new spline object.

        Parameters
        ----------
        states : SplineDataType
            Dictionary containing 'positions', 'radii'
        downsample_num_element : int | None, optional
            If provided, downsample the positions and radii to this number of elements.
            Default is None (no downsampling).
        """

        # TODO: Refactor this part: never copy-paste code. Make separate function in utils.py
        remaining_keys = set(states.keys()) - cls.input_states
        if len(remaining_keys) > 0:
            warnings.warn(
                f"{list(remaining_keys)} are not used as a part of the state definition."
            )
        return cls(
            states["positions"],
            states["radii"],
            downsample_num_element,
        )

    @property
    def material(self) -> bpy.types.Material:
        """
        Access the Blender material.
        """

        return self._material

    @property
    def object(self) -> bpy.types.Object:
        """
        Access the Blender object.
        """

        return self._obj

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the material of the sphere object.

        Parameters
        ----------
        color : NDArray
            The new color of the sphere object in RGBA format.
        """

        if "color" in kwargs:
            color = kwargs["color"]
            if isinstance(color, (tuple, list)):
                color = np.array(color)
            assert isinstance(
                color, np.ndarray
            ), "Keyword argument `color` should be a numpy array."
            assert color.shape == (
                4,
            ), "Keyword argument color should be a 1D array with 4 elements: RGBA."
            assert np.all(color >= 0) and np.all(
                color <= 1
            ), "Keyword argument color should be in the range of [0, 1]."
            self.material.diffuse_color = tuple(color)

    def update_states(
        self, positions: NDArray | None = None, radii: NDArray | None = None
    ) -> None:
        """
        Updates the position and radius of the spline object.

        Parameters
        ----------
        positioni : NDArray
            The new position of the spline object.
        radii : float
            The new radius of the spline object.

        Raises
        ------
        ValueError
            If the shape of the position or radius is incorrect, or if the data is NaN.
        """

        spline = self.object.data.splines[0]
        if positions is not None:
            _validate_position(positions)
            positions = self._downsample_data(
                positions, self.downsample_num_element
            )
            for i, point in enumerate(spline.bezier_points):
                x, y, z = positions[:, i]
                point.co = (x, y, z)
        if radii is not None:
            _validate_radii(radii)
            radii = self._downsample_data(
                radii.astype(float, copy=False), self.downsample_num_element
            )
            for i, point in enumerate(spline.bezier_points):
                point.radius = radii[i]

    def _downsample_data(
        self, vector: NDArray, num_elements: int | None
    ) -> NDArray:
        if num_elements is None:
            return vector
        if vector.shape[-1] <= num_elements:
            return vector

        t = np.linspace(0, 1, num_elements)
        t_old = np.linspace(0, 1, vector.shape[-1])
        if interp1d is not None:
            return interp1d(t_old, vector, axis=-1)(t)

        if vector.ndim == 1:
            return np.interp(t, t_old, vector)

        flattened = vector.reshape(-1, vector.shape[-1])
        downsampled = np.vstack([np.interp(t, t_old, row) for row in flattened])
        return downsampled.reshape(vector.shape[:-1] + (num_elements,))

    def _create_bezier_spline(
        self,
        number_of_points: int,
    ) -> bpy.types.Object:
        """
        Creates a new pipe object.

        Parameters
        ----------
        number_of_points : int
            The number of points in the pipe.
        """
        # Create a new curve
        curve_data = bpy.data.curves.new(name="spline_curve", type="CURVE")
        curve_data.dimensions = "3D"

        spline = curve_data.splines.new(type="BEZIER")
        spline.bezier_points.add(
            number_of_points - 1
        )  # First point is already there

        # Set the spline points and radii
        for i in range(number_of_points):
            point = spline.bezier_points[i]
            point.handle_left_type = point.handle_right_type = "FREE"

        # Create a new object with the curve data
        curve_object = bpy.data.objects.new("spline_curve_object", curve_data)
        curve_object.data.resolution_u = 1
        # curve_object.data.render_resolution_u = 1
        bpy.context.collection.objects.link(curve_object)

        # Create a bevel object for the pipe profile
        bpy.ops.curve.primitive_bezier_circle_add(
            radius=1,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        bevel_circle = bpy.context.object
        bevel_circle.name = "bevel_circle"
        # Set resolution for smoother bevel profile
        bevel_circle.data.resolution_u = 8
        # Hide the bevel circle object in the viewport and render
        bevel_circle.hide_viewport = True
        bevel_circle.hide_render = True

        # Set the bevel object to the curve
        curve_data.bevel_mode = "OBJECT"
        curve_data.bevel_object = bevel_circle
        curve_data.use_fill_caps = True

        return curve_object

    def update_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.

        Parameters
        ----------
        keyframe : int
        """
        spline = self.object.data.splines[0]

        for i, point in enumerate(spline.bezier_points):
            # This is to reset the left/right handle for each bezier curve points
            point.handle_left_type = "AUTO"
            point.handle_right_type = "AUTO"
            point.keyframe_insert(data_path="handle_left", frame=keyframe)
            point.keyframe_insert(data_path="handle_right", frame=keyframe)
            point.keyframe_insert(data_path="co", frame=keyframe)  # Coordinate
            point.keyframe_insert(data_path="radius", frame=keyframe)
        self.material.keyframe_insert(data_path="diffuse_color", frame=keyframe)


# ---------------------------------------------------------------------------
# Module-level helpers shared by BezierSplineFinnedPipe and BezierSplineAnnulusPipe
# ---------------------------------------------------------------------------


def _add_bezier_circle_to_curve(curve_data: bpy.types.Curve, r: float) -> None:
    """
    Append a closed 4-point Bezier circle spline of radius ``r`` to an
    existing curve data block.

    Uses handle factor k = r × 0.5523 (≈ (4/3)·tan(π/8)) for a near-exact
    circular approximation with < 0.027% radial error.
    """
    k = r * 0.5523
    s = curve_data.splines.new(type="BEZIER")
    s.bezier_points.add(3)  # 4 points total
    s.use_cyclic_u = True
    pts = [
        ((r, 0, 0), (r, -k, 0), (r, k, 0)),
        ((0, r, 0), (k, r, 0), (-k, r, 0)),
        ((-r, 0, 0), (-r, k, 0), (-r, -k, 0)),
        ((0, -r, 0), (-k, -r, 0), (k, -r, 0)),
    ]
    for i, (co, hl, hr) in enumerate(pts):
        pt = s.bezier_points[i]
        pt.co = co
        pt.handle_left = hl
        pt.handle_right = hr
        pt.handle_left_type = pt.handle_right_type = "FREE"


def _create_bevel_annulus(
    outer_radius: float, inner_radius: float
) -> bpy.types.Object:
    """
    Create a 2D bevel profile representing a hollow annulus.

    Two concentric Bezier circle splines in the same 2D curve. Blender's
    even-odd fill rule renders the region between them as filled and the
    inner circle as a hole.
    """
    curve_data = bpy.data.curves.new(name="bevel_annulus", type="CURVE")
    curve_data.dimensions = "2D"
    _add_bezier_circle_to_curve(curve_data, outer_radius)
    _add_bezier_circle_to_curve(curve_data, inner_radius)
    obj = bpy.data.objects.new("bevel_annulus", curve_data)
    bpy.context.collection.objects.link(obj)
    obj.hide_viewport = True
    obj.hide_render = True
    return obj


def _create_circle_spline(
    number_of_points: int,
    pipe_outer_radius: float,
    inner_to_outer_radius_ratio: float | None = None,
) -> bpy.types.Object:
    """
    Create a 3D Bezier spine curve with a circle (or annulus) bevel profile.

    When ``inner_to_outer_radius_ratio`` is provided, the bevel is an annulus
    (even-odd fill rule gives a hollow ring cross-section). Otherwise the
    bevel is a solid circle.

    Parameters
    ----------
    number_of_points : int
        Number of Bezier control points on the spine.
    pipe_outer_radius : float
        Outer radius of the circle / annulus bevel.
    inner_to_outer_radius_ratio : float or None, optional
        Ratio of inner to outer radius (0 < ratio < 1). Default is None (solid circle).
    """
    curve_data = bpy.data.curves.new(name="circle_spline_curve", type="CURVE")
    curve_data.dimensions = "3D"
    spline = curve_data.splines.new(type="BEZIER")
    spline.bezier_points.add(number_of_points - 1)
    for i in range(number_of_points):
        point = spline.bezier_points[i]
        point.handle_left_type = point.handle_right_type = "FREE"
    curve_object = bpy.data.objects.new(
        "circle_spline_curve_object", curve_data
    )
    curve_object.data.resolution_u = 1
    bpy.context.collection.objects.link(curve_object)

    if inner_to_outer_radius_ratio is not None:
        bevel_obj = _create_bevel_annulus(
            pipe_outer_radius,
            pipe_outer_radius * inner_to_outer_radius_ratio,
        )
    else:
        bpy.ops.curve.primitive_bezier_circle_add(
            radius=pipe_outer_radius,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
        )
        bevel_obj = bpy.context.object
        bevel_obj.name = "bevel_circle"
        bevel_obj.data.resolution_u = 8
        bevel_obj.hide_viewport = True
        bevel_obj.hide_render = True

    curve_data.bevel_mode = "OBJECT"
    curve_data.bevel_object = bevel_obj
    curve_data.use_fill_caps = True
    return curve_object


class BezierSplineFinnedPipe(KeyFrameControlMixin):
    """
    A compound Bezier spline primitive: a hollow annulus (or solid circle) pipe
    at the centerline with two rectangular fins offset along the d1 director.

    At construction two rectangular-cross-section fin spines are placed at
    ``positions ± node_r_rectangle_center * node_d1``.  An optional annulus
    (or solid circle) spine follows the centerline.  All three share the same
    ``dilatation``-driven ``point.radius`` scaling so volume is preserved for
    an incompressible rod.

    All inputs are expected to be **node-based** (shape (…, n_nodes)).

    Parameters
    ----------
    positions : NDArray
        Centerline control-point positions. Shape: (3, n_nodes).
    dilatation : NDArray
        Stretch ratio at each node. Shape: (n_nodes,).
        ``point.radius`` is set to ``1 / sqrt(dilatation)`` at every node.
    node_r_rectangle_center : NDArray
        Distance from the centerline to each fin's centre along d1.
        Shape: (n_nodes,).  Fixed at construction; not updated per frame.
    node_d1 : NDArray
        First-director unit vector at each node.  Shape: (3, n_nodes).
        Changes every frame; pass updated values to ``update_states``.
    half_fin_span : float
        Half-span (nominal width) of each rectangular fin cross-section (along local X).
    fin_thickness : float
        Nominal thickness (depth) of each rectangular fin cross-section (along local Y).
    pipe_outer_radius : float or None, optional
        Outer radius of the annulus/circle centerline bevel.
        If None, no circle spine is created.  Default is None.
    inner_to_outer_radius_ratio : float or None, optional
        When ``pipe_outer_radius`` is set, defines the hollow inner radius as
        ``inner_radius = pipe_outer_radius * inner_to_outer_radius_ratio``.
        Must be in (0, 1).  Default is None (solid circle).
    downsample_num_element : int or None, optional
        If provided, downsample spine control points to this number (min 2).
        Default is None (no downsampling).
    """

    input_states = {"positions", "dilatation", "node_d1"}
    name = "bspline_finned"

    def __init__(
        self,
        positions: NDArray,
        dilatation: NDArray,
        node_r_rectangle_center: NDArray,
        node_d1: NDArray,
        half_fin_span: float,
        fin_thickness: float,
        pipe_outer_radius: float | None = None,
        inner_to_outer_radius_ratio: float | None = None,
        downsample_num_element: int | None = None,
        **kwargs: Any,
    ) -> None:
        assert (
            downsample_num_element is None or downsample_num_element >= 2
        ), "downsample_num_element must be at least 2 to include both endpoints"
        self.downsample_num_element = downsample_num_element
        number_of_points = positions.shape[1]
        if downsample_num_element is not None:
            number_of_points = min(number_of_points, downsample_num_element)

        self._node_r_rectangle_center = node_r_rectangle_center

        # Two rectangular fin spines
        self._fin1_obj = self._create_rect_spine(
            number_of_points, half_fin_span, fin_thickness
        )
        self._fin1_material = bpy.data.materials.new(
            name=f"{self._fin1_obj.name}_material"
        )
        self._fin1_obj.data.materials.append(self._fin1_material)

        self._fin2_obj = self._create_rect_spine(
            number_of_points, half_fin_span, fin_thickness
        )
        self._fin2_material = bpy.data.materials.new(
            name=f"{self._fin2_obj.name}_material"
        )
        self._fin2_obj.data.materials.append(self._fin2_material)

        # Optional annulus / solid circle spine at the centerline
        self._circle_obj: bpy.types.Object | None = None
        self._circle_material: bpy.types.Material | None = None
        if pipe_outer_radius is not None:
            self._circle_obj = _create_circle_spline(
                number_of_points, pipe_outer_radius, inner_to_outer_radius_ratio
            )
            self._circle_material = bpy.data.materials.new(
                name=f"{self._circle_obj.name}_material"
            )
            self._circle_obj.data.materials.append(self._circle_material)

        self.update_states(positions, dilatation, node_d1)

    @classmethod
    def create(
        cls,
        states: SplineDataType,
        half_fin_span: float = 1.0,
        fin_thickness: float = 1.0,
        pipe_outer_radius: float | None = None,
        inner_to_outer_radius_ratio: float | None = None,
        downsample_num_element: int | None = None,
    ) -> "BezierSplineFinnedPipe":
        """
        Factory method. Expects node-based data in ``states``.

        Required keys: ``positions`` (3, n_nodes), ``dilatation`` (n_nodes,),
        ``node_r_rectangle_center`` (n_nodes,), ``node_d1`` (3, n_nodes).
        """
        remaining_keys = set(states.keys()) - {
            "positions",
            "dilatation",
            "node_r_rectangle_center",
            "node_d1",
        }
        if remaining_keys:
            warnings.warn(
                f"{list(remaining_keys)} are not used as a part of the state definition."
            )
        return cls(
            states["positions"],
            states["dilatation"],
            states["node_r_rectangle_center"],
            states["node_d1"],
            half_fin_span,
            fin_thickness,
            pipe_outer_radius,
            inner_to_outer_radius_ratio,
            downsample_num_element,
        )

    @property
    def material(self) -> dict[str, bpy.types.Material]:
        """Return a dict of all materials keyed by component name."""
        mats: dict[str, bpy.types.Material] = {
            "fin1": self._fin1_material,
            "fin2": self._fin2_material,
        }
        if self._circle_material is not None:
            mats["circle"] = self._circle_material
        return mats

    @property
    def object(self) -> dict[str, bpy.types.Object]:
        """Return a dict of all Blender objects keyed by component name."""
        objs: dict[str, bpy.types.Object] = {
            "fin1": self._fin1_obj,
            "fin2": self._fin2_obj,
        }
        if self._circle_obj is not None:
            objs["circle"] = self._circle_obj
        return objs

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Update the diffuse color of all component materials.

        Parameters
        ----------
        color : NDArray
            RGBA color array of shape (4,), values in [0, 1].
        """
        if "color" in kwargs:
            color = kwargs["color"]
            if isinstance(color, (tuple, list)):
                color = np.array(color)
            assert isinstance(
                color, np.ndarray
            ), "Keyword argument `color` should be a numpy array."
            assert color.shape == (
                4,
            ), "Keyword argument color should be a 1D array with 4 elements: RGBA."
            assert np.all(color >= 0) and np.all(
                color <= 1
            ), "Keyword argument color should be in the range of [0, 1]."
            self._fin1_material.diffuse_color = tuple(color)
            self._fin2_material.diffuse_color = tuple(color)
            if self._circle_material is not None:
                self._circle_material.diffuse_color = tuple(color)

    def update_states(
        self,
        positions: NDArray,
        dilatation: NDArray,
        node_d1: NDArray,
    ) -> None:
        """
        Update fin positions (computed from directors) and cross-section scale.

        Parameters
        ----------
        positions : NDArray
            Centerline node positions. Shape: (3, n_nodes).
        dilatation : NDArray
            Stretch ratio at each node. Shape: (n_nodes,).
        node_d1 : NDArray
            First director unit vector at each node. Shape: (3, n_nodes).
        """
        fin1_pos = (
            positions + self._node_r_rectangle_center[np.newaxis, :] * node_d1
        )
        fin2_pos = (
            positions - self._node_r_rectangle_center[np.newaxis, :] * node_d1
        )
        self._update_spine(self._fin1_obj, fin1_pos, dilatation)
        self._update_spine(self._fin2_obj, fin2_pos, dilatation)
        if self._circle_obj is not None:
            self._update_spine(self._circle_obj, positions, dilatation)

    def _update_spine(
        self,
        spine_obj: bpy.types.Object,
        positions: NDArray,
        dilatation: NDArray,
    ) -> None:
        spline = spine_obj.data.splines[0]
        _validate_position(positions)
        pos = self._downsample_data(positions, self.downsample_num_element)
        for i, point in enumerate(spline.bezier_points):
            x, y, z = pos[:, i]
            point.co = (x, y, z)
        _validate_radii(dilatation)
        dil = self._downsample_data(
            dilatation.astype(float, copy=False), self.downsample_num_element
        )
        for i, point in enumerate(spline.bezier_points):
            point.radius = 1.0 / np.sqrt(dil[i])

    def _downsample_data(
        self, vector: NDArray, num_elements: int | None
    ) -> NDArray:
        if num_elements is None:
            return vector
        if vector.shape[-1] <= num_elements:
            return vector

        t = np.linspace(0, 1, num_elements)
        t_old = np.linspace(0, 1, vector.shape[-1])
        if interp1d is not None:
            return interp1d(t_old, vector, axis=-1)(t)

        if vector.ndim == 1:
            return np.interp(t, t_old, vector)

        flattened = vector.reshape(-1, vector.shape[-1])
        downsampled = np.vstack([np.interp(t, t_old, row) for row in flattened])
        return downsampled.reshape(vector.shape[:-1] + (num_elements,))

    @staticmethod
    def _create_rect_spine(
        number_of_points: int,
        half_fin_span: float,
        fin_thickness: float,
    ) -> bpy.types.Object:
        """Create a Bezier spine curve with a rectangular bevel profile."""
        curve_data = bpy.data.curves.new(name="fin_spline_curve", type="CURVE")
        curve_data.dimensions = "3D"

        spline = curve_data.splines.new(type="BEZIER")
        spline.bezier_points.add(number_of_points - 1)

        for i in range(number_of_points):
            point = spline.bezier_points[i]
            point.handle_left_type = point.handle_right_type = "FREE"

        curve_object = bpy.data.objects.new(
            "fin_spline_curve_object", curve_data
        )
        curve_object.data.resolution_u = 1
        bpy.context.collection.objects.link(curve_object)

        bevel_rect = BezierSplineFinnedPipe._create_bevel_rectangle(
            half_fin_span, fin_thickness
        )
        curve_data.bevel_mode = "OBJECT"
        curve_data.bevel_object = bevel_rect
        curve_data.use_fill_caps = True

        return curve_object

    @staticmethod
    def _create_bevel_rectangle(
        half_fin_span: float, fin_thickness: float
    ) -> bpy.types.Object:
        """Create a closed rectangular 2D bevel profile curve."""
        curve_data = bpy.data.curves.new(name="bevel_rect", type="CURVE")
        curve_data.dimensions = "2D"

        rect_spline = curve_data.splines.new(type="BEZIER")
        rect_spline.bezier_points.add(3)  # 4 points total
        rect_spline.use_cyclic_u = True

        hw, hd = half_fin_span / 2.0, fin_thickness / 2.0
        corners = [(-hw, -hd, 0), (hw, -hd, 0), (hw, hd, 0), (-hw, hd, 0)]
        for i, (x, y, z) in enumerate(corners):
            pt = rect_spline.bezier_points[i]
            pt.co = (x, y, z)
            pt.handle_left_type = pt.handle_right_type = "VECTOR"

        obj = bpy.data.objects.new("bevel_rect", curve_data)
        bpy.context.collection.objects.link(obj)
        obj.hide_viewport = True
        obj.hide_render = True
        return obj

    def update_keyframe(self, keyframe: int) -> None:
        """Set keyframes for all spines and materials."""
        self._keyframe_spine(self._fin1_obj, keyframe)
        self._fin1_material.keyframe_insert(
            data_path="diffuse_color", frame=keyframe
        )
        self._keyframe_spine(self._fin2_obj, keyframe)
        self._fin2_material.keyframe_insert(
            data_path="diffuse_color", frame=keyframe
        )
        if self._circle_obj is not None:
            self._keyframe_spine(self._circle_obj, keyframe)
            self._circle_material.keyframe_insert(
                data_path="diffuse_color", frame=keyframe
            )

    @staticmethod
    def _keyframe_spine(spine_obj: bpy.types.Object, keyframe: int) -> None:
        spline = spine_obj.data.splines[0]
        for point in spline.bezier_points:
            point.handle_left_type = "AUTO"
            point.handle_right_type = "AUTO"
            point.keyframe_insert(data_path="handle_left", frame=keyframe)
            point.keyframe_insert(data_path="handle_right", frame=keyframe)
            point.keyframe_insert(data_path="co", frame=keyframe)
            point.keyframe_insert(data_path="radius", frame=keyframe)


class BezierSplineAnnulusPipe(KeyFrameControlMixin):
    """
    A single-spine Bezier pipe with a hollow annulus (or solid circle) cross-section.

    The cross-section shape is fixed at construction by ``pipe_outer_radius`` and
    (optionally) ``inner_to_outer_radius_ratio``. Per-point scaling is driven
    by ``dilatation`` — for an incompressible rod,
    ``point.radius = 1 / sqrt(dilatation)`` preserves cross-sectional area.

    Parameters
    ----------
    positions : NDArray
        Control-point positions. Shape: (3, n_nodes).
    dilatation : NDArray
        Stretch ratio at each node. Shape: (n_nodes,).
    pipe_outer_radius : float
        Outer radius of the annulus/circle bevel profile.
    inner_to_outer_radius_ratio : float or None, optional
        Ratio of inner to outer radius (0 < ratio < 1). When provided, a hollow
        annulus bevel is used. Default is None (solid circle).
    downsample_num_element : int or None, optional
        If provided, downsample spine control points to this number (min 2).
        Default is None (no downsampling).
    """

    input_states = {"positions", "dilatation"}
    name = "bspline_annulus"

    def __init__(
        self,
        positions: NDArray,
        dilatation: NDArray,
        pipe_outer_radius: float,
        inner_to_outer_radius_ratio: float | None = None,
        downsample_num_element: int | None = None,
        **kwargs: Any,
    ) -> None:
        assert (
            downsample_num_element is None or downsample_num_element >= 2
        ), "downsample_num_element must be at least 2 to include both endpoints"
        self.downsample_num_element = downsample_num_element
        number_of_points = positions.shape[1]
        if downsample_num_element is not None:
            number_of_points = min(number_of_points, downsample_num_element)

        self._obj = _create_circle_spline(
            number_of_points, pipe_outer_radius, inner_to_outer_radius_ratio
        )
        self._obj.name = self.name
        self._material = bpy.data.materials.new(
            name=f"{self._obj.name}_material"
        )
        self._obj.data.materials.append(self._material)

        self.update_states(positions, dilatation)

    @classmethod
    def create(
        cls,
        states: SplineDataType,
        pipe_outer_radius: float = 1.0,
        inner_to_outer_radius_ratio: float | None = None,
        downsample_num_element: int | None = None,
    ) -> "BezierSplineAnnulusPipe":
        """
        Factory method to create a new BezierSplineAnnulusPipe.

        Parameters
        ----------
        states : SplineDataType
            Must contain keys: ``positions`` (3, n_nodes), ``dilatation`` (n_nodes,).
            May also contain ``pipe_outer_radius`` and ``inner_to_outer_radius_ratio``
            as scalars; values found in ``states`` take precedence over keyword defaults.
        pipe_outer_radius : float, optional
            Outer radius of the annulus/circle bevel. Default is 1.0.
        inner_to_outer_radius_ratio : float or None, optional
            Ratio for hollow inner radius. Default is None (solid circle).
        downsample_num_element : int or None, optional
            Downsample control points to this number. Default is None.
        """
        remaining_keys = set(states.keys()) - {
            "positions",
            "dilatation",
            "pipe_outer_radius",
            "inner_to_outer_radius_ratio",
        }
        if remaining_keys:
            warnings.warn(
                f"{list(remaining_keys)} are not used as a part of the state definition."
            )
        _radius = (
            float(states["pipe_outer_radius"])
            if "pipe_outer_radius" in states
            else pipe_outer_radius
        )
        _ratio = (
            float(states["inner_to_outer_radius_ratio"])
            if "inner_to_outer_radius_ratio" in states
            else inner_to_outer_radius_ratio
        )
        return cls(
            states["positions"],
            states["dilatation"],
            _radius,
            _ratio,
            downsample_num_element,
        )

    @property
    def material(self) -> bpy.types.Material:
        """Access the Blender material."""
        return self._material

    @property
    def object(self) -> bpy.types.Object:
        """Access the Blender object."""
        return self._obj

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Update the diffuse color of the material.

        Parameters
        ----------
        color : NDArray
            RGBA color array of shape (4,), values in [0, 1].
        """
        if "color" in kwargs:
            color = kwargs["color"]
            if isinstance(color, (tuple, list)):
                color = np.array(color)
            assert isinstance(
                color, np.ndarray
            ), "Keyword argument `color` should be a numpy array."
            assert color.shape == (
                4,
            ), "Keyword argument color should be a 1D array with 4 elements: RGBA."
            assert np.all(color >= 0) and np.all(
                color <= 1
            ), "Keyword argument color should be in the range of [0, 1]."
            self._material.diffuse_color = tuple(color)

    def update_states(
        self,
        positions: NDArray | None = None,
        dilatation: NDArray | None = None,
    ) -> None:
        """
        Update control-point positions and cross-section scale.

        Parameters
        ----------
        positions : NDArray, optional
            New control-point positions. Shape: (3, n_nodes).
        dilatation : NDArray, optional
            Stretch ratio at each node. Shape: (n_nodes,).
            Sets ``point.radius = 1 / sqrt(dilatation)`` at each node.
        """
        spline = self._obj.data.splines[0]
        if positions is not None:
            _validate_position(positions)
            pos = self._downsample_data(positions, self.downsample_num_element)
            for i, point in enumerate(spline.bezier_points):
                x, y, z = pos[:, i]
                point.co = (x, y, z)
        if dilatation is not None:
            _validate_radii(dilatation)
            dil = self._downsample_data(
                dilatation.astype(float, copy=False),
                self.downsample_num_element,
            )
            for i, point in enumerate(spline.bezier_points):
                point.radius = 1.0 / np.sqrt(dil[i])

    def _downsample_data(
        self, vector: NDArray, num_elements: int | None
    ) -> NDArray:
        if num_elements is None:
            return vector
        if vector.shape[-1] <= num_elements:
            return vector
        t = np.linspace(0, 1, num_elements)
        t_old = np.linspace(0, 1, vector.shape[-1])
        if interp1d is not None:
            return interp1d(t_old, vector, axis=-1)(t)
        if vector.ndim == 1:
            return np.interp(t, t_old, vector)
        flattened = vector.reshape(-1, vector.shape[-1])
        downsampled = np.vstack([np.interp(t, t_old, row) for row in flattened])
        return downsampled.reshape(vector.shape[:-1] + (num_elements,))

    def update_keyframe(self, keyframe: int) -> None:
        """Set keyframe for the annulus spine and material."""
        spline = self._obj.data.splines[0]
        for point in spline.bezier_points:
            point.handle_left_type = "AUTO"
            point.handle_right_type = "AUTO"
            point.keyframe_insert(data_path="handle_left", frame=keyframe)
            point.keyframe_insert(data_path="handle_right", frame=keyframe)
            point.keyframe_insert(data_path="co", frame=keyframe)
            point.keyframe_insert(data_path="radius", frame=keyframe)
        self._material.keyframe_insert(
            data_path="diffuse_color", frame=keyframe
        )


if TYPE_CHECKING:
    # This is required for explicit type-checking
    data = {
        "positions": np.array([[0, 0, 0], [1, 1, 1]]).T,
        "radii": np.array([1.0, 1.0]),
    }
    _: BlenderMeshInterfaceProtocol = BezierSplinePipe.create(data)
