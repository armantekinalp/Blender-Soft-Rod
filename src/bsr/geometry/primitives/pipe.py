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
    "BezierSplineRectAnnulusPipe",
    "BezierSplineFinSegmentPipe",
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


def _create_bevel_rect_annulus(
    rect_width: float,
    rect_depth: float,
    bore_radius: float,
) -> bpy.types.Object:
    """
    Create a 2D bevel profile: filled rectangle with a circular bore.

    A single 2D curve contains two splines:
    1. Outer rectangle — 4-point Bezier with VECTOR handles.
    2. Inner circle of radius ``bore_radius`` — 4-point Bezier approximation.

    Blender's even-odd fill rule punches the circle out of the rectangle.

    Parameters
    ----------
    rect_width : float
        Full width of the outer rectangle.
    rect_depth : float
        Full depth of the outer rectangle.
    bore_radius : float
        Radius of the circular bore. Must be < min(rect_width, rect_depth) / 2.
    """
    assert bore_radius < min(rect_width, rect_depth) / 2.0, (
        f"bore_radius ({bore_radius}) must be smaller than "
        f"min(rect_width, rect_depth) / 2 = {min(rect_width, rect_depth) / 2.0}"
    )
    curve_data = bpy.data.curves.new(name="bevel_rect_annulus", type="CURVE")
    curve_data.dimensions = "2D"

    # Outer rectangle
    rect_spline = curve_data.splines.new(type="BEZIER")
    rect_spline.bezier_points.add(3)  # 4 points total
    rect_spline.use_cyclic_u = True
    hw, hd = rect_width / 2.0, rect_depth / 2.0
    corners = [(-hw, -hd, 0), (hw, -hd, 0), (hw, hd, 0), (-hw, hd, 0)]
    for i, (x, y, z) in enumerate(corners):
        pt = rect_spline.bezier_points[i]
        pt.co = (x, y, z)
        pt.handle_left_type = pt.handle_right_type = "VECTOR"

    # Inner circular bore
    _add_bezier_circle_to_curve(curve_data, bore_radius)

    obj = bpy.data.objects.new("bevel_rect_annulus", curve_data)
    bpy.context.collection.objects.link(obj)
    obj.hide_viewport = True
    obj.hide_render = True
    return obj


def _create_bevel_rectangle_asymmetric(
    left_span: float, right_span: float, fin_thickness: float
) -> bpy.types.Object:
    """
    Create a closed rectangular 2D bevel profile curve whose two edges along
    the span axis are independently sized relative to the local origin
    (the spine), instead of being symmetric about it.

    The rectangle spans from ``-left_span`` to ``right_span`` along local Y
    (the span axis) and ``±fin_thickness / 2`` along local X (the thickness
    axis) — the same axis convention as
    ``BezierSplineFinnedPipe._create_bevel_rectangle``, generalized to
    independent edges so the spine can sit anywhere within (or at the edge
    of) the rectangle rather than always at its center.
    """
    curve_data = bpy.data.curves.new(
        name="bevel_rect_asymmetric", type="CURVE"
    )
    curve_data.dimensions = "2D"

    rect_spline = curve_data.splines.new(type="BEZIER")
    rect_spline.bezier_points.add(3)  # 4 points total
    rect_spline.use_cyclic_u = True

    hw = fin_thickness / 2.0
    corners = [
        (-hw, -left_span, 0),
        (hw, -left_span, 0),
        (hw, right_span, 0),
        (-hw, right_span, 0),
    ]
    for i, (x, y, z) in enumerate(corners):
        pt = rect_spline.bezier_points[i]
        pt.co = (x, y, z)
        pt.handle_left_type = pt.handle_right_type = "VECTOR"

    obj = bpy.data.objects.new("bevel_rect_asymmetric", curve_data)
    bpy.context.collection.objects.link(obj)
    obj.hide_viewport = True
    obj.hide_render = True
    return obj


def _create_per_element_asymmetric_rect_spine(
    left_span: NDArray,
    right_span: NDArray,
    fin_thickness: float,
) -> tuple[list, list]:
    """
    Create one 2-point Bezier curve per element, each with its own
    asymmetric rectangular bevel (``_create_bevel_rectangle_asymmetric``).

    Parameters
    ----------
    left_span : NDArray
        Per-element distance from the spine to the "left" edge. Shape: (n_elems,).
    right_span : NDArray
        Per-element distance from the spine to the "right" edge. Shape: (n_elems,).
    fin_thickness : float
        Thickness shared across all elements.

    Returns
    -------
    tuple of (list of Objects, list of Materials)
    """
    objects = []
    materials = []
    for l_span, r_span in zip(left_span, right_span):
        curve_data = bpy.data.curves.new(
            name="fin_segment_spline_curve", type="CURVE"
        )
        curve_data.dimensions = "3D"
        spline = curve_data.splines.new(type="BEZIER")
        spline.bezier_points.add(1)  # 2 points total
        for point in spline.bezier_points:
            point.handle_left_type = point.handle_right_type = "FREE"

        curve_obj = bpy.data.objects.new(
            "fin_segment_spline_curve_object", curve_data
        )
        curve_obj.data.resolution_u = 1
        bpy.context.collection.objects.link(curve_obj)

        bevel_rect = _create_bevel_rectangle_asymmetric(
            float(l_span), float(r_span), fin_thickness
        )
        curve_data.bevel_mode = "OBJECT"
        curve_data.bevel_object = bevel_rect
        curve_data.use_fill_caps = True

        mat = bpy.data.materials.new(name=f"{curve_obj.name}_material")
        curve_obj.data.materials.append(mat)

        objects.append(curve_obj)
        materials.append(mat)
    return objects, materials


def _create_rect_annulus_spline(
    number_of_points: int,
    rect_width: float,
    rect_depth: float,
    bore_radius: float,
) -> bpy.types.Object:
    """
    Create a 3D Bezier spine curve with a rect-annulus bevel profile.

    Parameters
    ----------
    number_of_points : int
        Number of Bezier control points on the spine.
    rect_width : float
        Full width of the outer rectangle cross-section.
    rect_depth : float
        Full depth of the outer rectangle cross-section.
    bore_radius : float
        Radius of the circular bore through the center.
    """
    curve_data = bpy.data.curves.new(
        name="rect_annulus_spline_curve", type="CURVE"
    )
    curve_data.dimensions = "3D"
    spline = curve_data.splines.new(type="BEZIER")
    spline.bezier_points.add(number_of_points - 1)
    for i in range(number_of_points):
        point = spline.bezier_points[i]
        point.handle_left_type = point.handle_right_type = "FREE"
    curve_object = bpy.data.objects.new(
        "rect_annulus_spline_curve_object", curve_data
    )
    curve_object.data.resolution_u = 1
    bpy.context.collection.objects.link(curve_object)

    bevel_obj = _create_bevel_rect_annulus(rect_width, rect_depth, bore_radius)
    curve_data.bevel_mode = "OBJECT"
    curve_data.bevel_object = bevel_obj
    curve_data.use_fill_caps = True
    return curve_object


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
    half_fin_span : float or NDArray
        Half-span (nominal width) of each rectangular fin cross-section (along local X).
        When a scalar ``float``, the same span is used for the entire rod.
        When a 1-D ``NDArray`` of shape ``(n_elems,)``, each element gets its
        own bevel rectangle with the corresponding span, enabling spatially
        varying fin geometry along the rod.
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
        Only used for the uniform-span (scalar) path.
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
        half_fin_span: float | NDArray,
        fin_thickness: float,
        pipe_outer_radius: float | None = None,
        inner_to_outer_radius_ratio: float | None = None,
        downsample_num_element: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._node_r_rectangle_center = node_r_rectangle_center
        self._varying_span = np.ndim(half_fin_span) > 0

        if self._varying_span:
            half_fin_span_arr = np.asarray(half_fin_span, dtype=float)
            assert (
                half_fin_span_arr.ndim == 1
            ), "half_fin_span array must be 1-D with shape (n_elems,)"
            n_elems = positions.shape[1] - 1
            assert len(half_fin_span_arr) == n_elems, (
                f"half_fin_span array length ({len(half_fin_span_arr)}) must equal "
                f"n_elems ({n_elems})"
            )
            self._fin1_objs, self._fin1_materials = (
                self._create_per_element_rect_spine(
                    half_fin_span_arr, fin_thickness
                )
            )
            self._fin2_objs, self._fin2_materials = (
                self._create_per_element_rect_spine(
                    half_fin_span_arr, fin_thickness
                )
            )
            # Aliases used by shared helpers (material property, update_material)
            self._fin1_obj = None
            self._fin2_obj = None
            self._fin1_material = None
            self._fin2_material = None
        else:
            assert (
                downsample_num_element is None or downsample_num_element >= 2
            ), "downsample_num_element must be at least 2 to include both endpoints"
            number_of_points = positions.shape[1]
            if downsample_num_element is not None:
                number_of_points = min(number_of_points, downsample_num_element)

            self._fin1_obj = self._create_rect_spine(
                number_of_points, float(half_fin_span), fin_thickness
            )
            self._fin1_material = bpy.data.materials.new(
                name=f"{self._fin1_obj.name}_material"
            )
            self._fin1_obj.data.materials.append(self._fin1_material)

            self._fin2_obj = self._create_rect_spine(
                number_of_points, float(half_fin_span), fin_thickness
            )
            self._fin2_material = bpy.data.materials.new(
                name=f"{self._fin2_obj.name}_material"
            )
            self._fin2_obj.data.materials.append(self._fin2_material)

            self._fin1_objs = []
            self._fin2_objs = []
            self._fin1_materials = []
            self._fin2_materials = []

        self.downsample_num_element = downsample_num_element

        # Optional annulus / solid circle spine at the centerline
        self._circle_obj: bpy.types.Object | None = None
        self._circle_material: bpy.types.Material | None = None
        if pipe_outer_radius is not None:
            if self._varying_span:
                n_pts = positions.shape[1]
            else:
                n_pts = positions.shape[1]
                if downsample_num_element is not None:
                    n_pts = min(n_pts, downsample_num_element)
            self._circle_obj = _create_circle_spline(
                n_pts, pipe_outer_radius, inner_to_outer_radius_ratio
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
        half_fin_span: float | NDArray = 1.0,
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
        if self._varying_span:
            mats: dict[str, bpy.types.Material] = {}
            for i, m in enumerate(self._fin1_materials):
                mats[f"fin1_{i}"] = m
            for i, m in enumerate(self._fin2_materials):
                mats[f"fin2_{i}"] = m
        else:
            mats = {
                "fin1": self._fin1_material,
                "fin2": self._fin2_material,
            }
        if self._circle_material is not None:
            mats["circle"] = self._circle_material
        return mats

    @property
    def object(self) -> dict[str, bpy.types.Object]:
        """Return a dict of all Blender objects keyed by component name."""
        if self._varying_span:
            objs: dict[str, bpy.types.Object] = {}
            for i, o in enumerate(self._fin1_objs):
                objs[f"fin1_{i}"] = o
            for i, o in enumerate(self._fin2_objs):
                objs[f"fin2_{i}"] = o
        else:
            objs = {
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
            color_tuple = tuple(color)
            if self._varying_span:
                for m in self._fin1_materials:
                    m.diffuse_color = color_tuple
                for m in self._fin2_materials:
                    m.diffuse_color = color_tuple
            else:
                self._fin1_material.diffuse_color = color_tuple
                self._fin2_material.diffuse_color = color_tuple
            if self._circle_material is not None:
                self._circle_material.diffuse_color = color_tuple

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
        if self._varying_span:
            self._update_per_element_fins(self._fin1_objs, fin1_pos, dilatation)
            self._update_per_element_fins(self._fin2_objs, fin2_pos, dilatation)
        else:
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

    @staticmethod
    def _update_per_element_fins(
        fin_objs: list,
        fin_positions: NDArray,
        dilatation: NDArray,
    ) -> None:
        """Update per-element fin curves (varying-span path).

        Parameters
        ----------
        fin_objs : list of bpy.types.Object
            One 2-point Bezier curve per element.
        fin_positions : NDArray
            Node-based fin spine positions. Shape: (3, n_nodes).
        dilatation : NDArray
            Node-based dilatation. Shape: (n_nodes,).
        """
        _validate_position(fin_positions)
        _validate_radii(dilatation)
        for elem_idx, obj in enumerate(fin_objs):
            spline = obj.data.splines[0]
            for local_pt_idx, node_idx in enumerate([elem_idx, elem_idx + 1]):
                pt = spline.bezier_points[local_pt_idx]
                x, y, z = fin_positions[:, node_idx]
                pt.co = (x, y, z)
                pt.radius = 1.0 / np.sqrt(float(dilatation[node_idx]))

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
    def _create_per_element_rect_spine(
        half_fin_span: NDArray,
        fin_thickness: float,
    ) -> tuple[list, list]:
        """Create one 2-point Bezier curve per element, each with its own bevel.

        Parameters
        ----------
        half_fin_span : NDArray
            Per-element half-span values. Shape: (n_elems,).
        fin_thickness : float
            Fin thickness shared across all elements.

        Returns
        -------
        tuple of (list of Objects, list of Materials)
        """
        objects = []
        materials = []
        for span in half_fin_span:
            curve_data = bpy.data.curves.new(
                name="fin_spline_curve", type="CURVE"
            )
            curve_data.dimensions = "3D"
            spline = curve_data.splines.new(type="BEZIER")
            spline.bezier_points.add(1)  # 2 points total
            for point in spline.bezier_points:
                point.handle_left_type = point.handle_right_type = "FREE"

            curve_obj = bpy.data.objects.new(
                "fin_spline_curve_object", curve_data
            )
            curve_obj.data.resolution_u = 1
            bpy.context.collection.objects.link(curve_obj)

            bevel_rect = BezierSplineFinnedPipe._create_bevel_rectangle(
                float(span), fin_thickness
            )
            curve_data.bevel_mode = "OBJECT"
            curve_data.bevel_object = bevel_rect
            curve_data.use_fill_caps = True

            mat = bpy.data.materials.new(name=f"{curve_obj.name}_material")
            curve_obj.data.materials.append(mat)

            objects.append(curve_obj)
            materials.append(mat)
        return objects, materials

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

        hw, hd = fin_thickness / 2.0, half_fin_span / 2.0
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
        if self._varying_span:
            for obj, mat in zip(self._fin1_objs, self._fin1_materials):
                self._keyframe_spine(obj, keyframe)
                mat.keyframe_insert(data_path="diffuse_color", frame=keyframe)
            for obj, mat in zip(self._fin2_objs, self._fin2_materials):
                self._keyframe_spine(obj, keyframe)
                mat.keyframe_insert(data_path="diffuse_color", frame=keyframe)
        else:
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


class BezierSplineRectAnnulusPipe(KeyFrameControlMixin):
    """
    A single-spine Bezier pipe with a rectangle-with-circular-bore cross-section.

    The bevel profile is a filled rectangle (``rect_width`` × ``rect_depth``) with a
    circular bore of radius ``bore_radius`` punched through its center, giving the
    appearance of a rectangular tube with a round internal bore.

    Cross-section geometry is fixed at construction. Per-frame updates drive spine
    control-point positions and per-node radius scaling
    (``point.radius = 1 / sqrt(dilatation)`` for an incompressible rod).

    Parameters
    ----------
    positions : NDArray
        Control-point positions. Shape: (3, n_nodes).
    dilatation : NDArray
        Stretch ratio at each node. Shape: (n_nodes,).
    rect_width : float
        Full width of the outer rectangle cross-section.
    rect_depth : float
        Full depth of the outer rectangle cross-section.
    bore_radius : float
        Radius of the circular bore through the center.
        Must be < min(rect_width, rect_depth) / 2.
    downsample_num_element : int or None, optional
        Downsample spine control points to this number (min 2). Default is None.
    """

    input_states = {"positions", "dilatation"}
    name = "bspline_rect_annulus"

    def __init__(
        self,
        positions: NDArray,
        dilatation: NDArray,
        rect_width: float,
        rect_depth: float,
        bore_radius: float,
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

        self._obj = _create_rect_annulus_spline(
            number_of_points, rect_width, rect_depth, bore_radius
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
        rect_width: float = 1.0,
        rect_depth: float = 1.0,
        bore_radius: float = 0.25,
        downsample_num_element: int | None = None,
    ) -> "BezierSplineRectAnnulusPipe":
        """
        Factory method to create a new BezierSplineRectAnnulusPipe.

        Parameters
        ----------
        states : SplineDataType
            Must contain: ``positions`` (3, n_nodes), ``dilatation`` (n_nodes,).
            May also contain ``rect_width``, ``rect_depth``, and ``bore_radius``
            as scalars; values found in ``states`` take precedence over keyword defaults.
        rect_width : float, optional
            Full width of the outer rectangle. Default is 1.0.
        rect_depth : float, optional
            Full depth of the outer rectangle. Default is 1.0.
        bore_radius : float, optional
            Radius of the circular bore. Default is 0.25.
        downsample_num_element : int or None, optional
            Downsample control points. Default is None.
        """
        remaining_keys = set(states.keys()) - {
            "positions",
            "dilatation",
            "rect_width",
            "rect_depth",
            "bore_radius",
        }
        if remaining_keys:
            warnings.warn(
                f"{list(remaining_keys)} are not used as a part of the state definition."
            )
        _rect_width = (
            float(states["rect_width"])
            if "rect_width" in states
            else rect_width
        )
        _rect_depth = (
            float(states["rect_depth"])
            if "rect_depth" in states
            else rect_depth
        )
        _bore_radius = (
            float(states["bore_radius"])
            if "bore_radius" in states
            else bore_radius
        )
        return cls(
            states["positions"],
            states["dilatation"],
            _rect_width,
            _rect_depth,
            _bore_radius,
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
        """Set keyframe for the rect-annulus spine and material."""
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


class BezierSplineFinSegmentPipe(KeyFrameControlMixin):
    """
    A single rectangular-cross-section fin blade swept along an arbitrary,
    already-resolved spine, with independently varying left/right edges.

    Unlike ``BezierSplineFinnedPipe`` (two fixed, symmetric fins offset from
    a rod centerline by a single distance), this primitive renders exactly
    one blade along a spine that the caller has already fully resolved
    (e.g. a rod's own centerline, or an externally-computed offset path
    derived from the rod's directors). The cross-section is a rectangle
    whose two edges relative to the spine -- ``left_span`` and
    ``right_span`` -- vary independently per element, which lets a blade
    taper to a point at one edge (e.g. flush against a rod's centerline)
    while the other edge tracks an adjacent pipe boundary, without any
    boolean/CSG cross-section work.

    All inputs are node-based (shape (..., n_nodes)) except ``left_span``
    and ``right_span``, which are per-element (shape (n_elems,)) since the
    cross-section is fixed within each element.

    Parameters
    ----------
    positions : NDArray
        Spine control-point positions, already resolved by the caller.
        Shape: (3, n_nodes).
    dilatation : NDArray
        Stretch ratio at each node. Shape: (n_nodes,).
        ``point.radius`` is set to ``1 / sqrt(dilatation)`` at every node.
    left_span : NDArray
        Per-element distance from the spine to the "left" edge of the
        rectangle. Shape: (n_elems,). Fixed at construction; not updated
        per frame.
    right_span : NDArray
        Per-element distance from the spine to the "right" edge of the
        rectangle. Shape: (n_elems,). Fixed at construction; not updated
        per frame.
    fin_thickness : float
        Thickness of the cross-section, perpendicular to the span axis.
        Fixed at construction; not updated per frame.
    """

    input_states = {"positions", "dilatation"}
    name = "bspline_fin_segment"

    def __init__(
        self,
        positions: NDArray,
        dilatation: NDArray,
        left_span: NDArray,
        right_span: NDArray,
        fin_thickness: float,
        **kwargs: Any,
    ) -> None:
        left_span_arr = np.asarray(left_span, dtype=float)
        right_span_arr = np.asarray(right_span, dtype=float)
        assert (
            left_span_arr.ndim == 1 and right_span_arr.ndim == 1
        ), "left_span and right_span must be 1-D arrays with shape (n_elems,)"
        n_elems = positions.shape[1] - 1
        assert len(left_span_arr) == n_elems and len(right_span_arr) == n_elems, (
            f"left_span/right_span length ({len(left_span_arr)}, "
            f"{len(right_span_arr)}) must equal n_elems ({n_elems})"
        )

        self._segment_objs, self._segment_materials = (
            _create_per_element_asymmetric_rect_spine(
                left_span_arr, right_span_arr, fin_thickness
            )
        )

        self.update_states(positions, dilatation)

    @classmethod
    def create(
        cls,
        states: SplineDataType,
        left_span: NDArray | None = None,
        right_span: NDArray | None = None,
        fin_thickness: float | None = None,
    ) -> "BezierSplineFinSegmentPipe":
        """
        Factory method. Expects node-based ``positions``/``dilatation`` and
        per-element ``left_span``/``right_span`` in ``states``.

        Required keys: ``positions`` (3, n_nodes), ``dilatation`` (n_nodes,).
        ``left_span``, ``right_span``, and ``fin_thickness`` may be supplied
        either as keyword arguments or inside ``states``; values found in
        ``states`` take precedence.
        """
        remaining_keys = set(states.keys()) - {
            "positions",
            "dilatation",
            "left_span",
            "right_span",
            "fin_thickness",
        }
        if remaining_keys:
            warnings.warn(
                f"{list(remaining_keys)} are not used as a part of the state definition."
            )
        _left_span = (
            np.asarray(states["left_span"])
            if "left_span" in states
            else left_span
        )
        _right_span = (
            np.asarray(states["right_span"])
            if "right_span" in states
            else right_span
        )
        _fin_thickness = (
            float(states["fin_thickness"])
            if "fin_thickness" in states
            else fin_thickness
        )
        assert (
            _left_span is not None
            and _right_span is not None
            and _fin_thickness is not None
        ), (
            "left_span, right_span, and fin_thickness must be provided "
            "either as keyword arguments or inside `states`."
        )
        return cls(
            states["positions"],
            states["dilatation"],
            _left_span,
            _right_span,
            _fin_thickness,
        )

    @property
    def material(self) -> dict[str, bpy.types.Material]:
        """Return a dict of all materials keyed by element index."""
        return {
            f"segment_{i}": m for i, m in enumerate(self._segment_materials)
        }

    @property
    def object(self) -> dict[str, bpy.types.Object]:
        """Return a dict of all Blender objects keyed by element index."""
        return {f"segment_{i}": o for i, o in enumerate(self._segment_objs)}

    def update_material(self, **kwargs: Any) -> None:
        """
        Update the diffuse color of all segment materials.

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
            color_tuple = tuple(color)
            for m in self._segment_materials:
                m.diffuse_color = color_tuple

    def update_states(
        self,
        positions: NDArray,
        dilatation: NDArray,
    ) -> None:
        """
        Update segment spine positions and cross-section scale.

        Parameters
        ----------
        positions : NDArray
            Spine node positions. Shape: (3, n_nodes).
        dilatation : NDArray
            Stretch ratio at each node. Shape: (n_nodes,).
        """
        _validate_position(positions)
        _validate_radii(dilatation)
        for elem_idx, obj in enumerate(self._segment_objs):
            spline = obj.data.splines[0]
            for local_pt_idx, node_idx in enumerate([elem_idx, elem_idx + 1]):
                pt = spline.bezier_points[local_pt_idx]
                x, y, z = positions[:, node_idx]
                pt.co = (x, y, z)
                pt.radius = 1.0 / np.sqrt(float(dilatation[node_idx]))

    def update_keyframe(self, keyframe: int) -> None:
        """Set keyframes for all segment spines and materials."""
        for obj, mat in zip(self._segment_objs, self._segment_materials):
            self._keyframe_spine(obj, keyframe)
            mat.keyframe_insert(data_path="diffuse_color", frame=keyframe)

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


if TYPE_CHECKING:
    # This is required for explicit type-checking
    data = {
        "positions": np.array([[0, 0, 0], [1, 1, 1]]).T,
        "radii": np.array([1.0, 1.0]),
    }
    _: BlenderMeshInterfaceProtocol = BezierSplinePipe.create(data)
