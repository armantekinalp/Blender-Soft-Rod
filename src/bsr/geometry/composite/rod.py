__doc__ = """
Rod class for creating and updating rods in Blender
"""
__all__ = [
    "RodWithSphereAndCylinder",
    "Rod",
    "RodWithCylinder",
    "RodWithBox",
    "RodWithSpline",
    "FinnedRodWithSpline",
    "AnnulusRodWithSpline",
    "RectAnnulusRodWithSpline",
]

from typing import TYPE_CHECKING, Any

from collections import defaultdict

import bpy
import numpy as np
from numpy.typing import NDArray

from bsr.geometry.primitives.pipe import (
    BezierSplineAnnulusPipe,
    BezierSplineFinnedPipe,
    BezierSplinePipe,
    BezierSplineRectAnnulusPipe,
)
from bsr.geometry.primitives.simple import Box, Cylinder, Sphere
from bsr.geometry.protocol import CompositeProtocol
from bsr.tools.keyframe_mixin import KeyFrameControlMixin


class RodWithSphereAndCylinder(KeyFrameControlMixin):
    """
    This class provides a mesh interface for Rod objects.
    Rod objects are created using given positions and radii.

    Parameters
    ----------
    positions : NDArray
        The positions of the Rod objects. Expected shape is (n_dim, n_nodes).
        n_dim = 3
    radii : NDArray
        The radii of the Rod objects. Expected shape is (n_nodes-1,).
    """

    input_states = {"positions", "radii"}

    def __init__(self, positions: NDArray, radii: NDArray) -> None:
        """
        Rod class constructor
        """
        # create sphere and cylinder objects
        self.spheres: list[Sphere] = []
        self.cylinders: list[Cylinder] = []
        self._bpy_objs: dict[str, list[bpy.types.Object]] = {
            "sphere": self.spheres,
            "cylinder": self.cylinders,
        }

        # create sphere and cylinder materials self.spheres_material: list[bpy.types.Material] = []
        self.spheres_material: list[bpy.types.Material] = []
        self.cylinders_material: list[bpy.types.Material] = []
        self._bpy_materials: dict[str, list[bpy.types.Material]] = {
            "sphere": self.spheres_material,
            "cylinder": self.cylinders_material,
        }

        self._build(positions, radii)

    @property
    def material(self) -> dict[str, list[bpy.types.Material]]:
        """
        Return the dictionary of Blender materials: sphere and cylinder
        """
        return self._bpy_materials

    @property
    def object(self) -> dict[str, list[bpy.types.Object]]:
        """
        Return the dictionary of Blender objects: sphere and cylinder
        """
        return self._bpy_objs

    @classmethod
    def create(cls, states: dict[str, NDArray]) -> "RodWithSphereAndCylinder":
        """
        Basic factory method to create a new Rod object.
        States must have the following keys: positions(n_dim, n_nodes), radii(n_nodes-1,)

        Parameters
        ----------
        states: dict[str, NDArray]
            A dictionary where keys are state names and values are NDArrays.

        Returns
        -------
        RodWithSphereAndCylinder
            An object of Rod class containing the predefined states
        """
        positions = states["positions"]
        radii = states["radii"]
        rod = cls(positions, radii)
        return rod

    def _build(self, positions: NDArray, radii: NDArray) -> None:
        """
        Populates the positions and radii of the Spheres and Cylinders into Rod object

        Parameters
        ----------
        positions: NDArray
            An array of shape (n_dim, n_nodes) that stores the positions of Spheres and Cylinders
        radii: NDArray
            An array of shape (n_nodes-1,) that stores the radii of the Spheres and Cylinders
        """
        _radii = np.concatenate([radii, [0]])
        _radii[1:] += radii
        _radii[1:-1] /= 2.0
        for j in range(positions.shape[-1]):
            sphere = Sphere(positions[:, j], _radii[j])
            self.spheres.append(sphere)
            self.spheres_material.append(sphere.material)

        for j in range(radii.shape[-1]):
            cylinder = Cylinder(
                positions[:, j],
                positions[:, j + 1],
                radii[j],
            )
            self.cylinders.append(cylinder)
            self.cylinders_material.append(cylinder.material)

    def update_states(self, positions: NDArray, radii: NDArray) -> None:
        """
        Update the states of the Rod object

        Parameters
        ----------
        positions : NDArray
            The positions of the Rod objects. Expected shape is (n_dim, n_nodes)
        radii : NDArray
            The radii of the Rod objects. Expected shape is (n_nodes-1,)
        """
        # check shape of positions and radii
        assert positions.ndim == 2, "positions must be 2D array"
        assert positions.shape[0] == 3, "positions must have 3 rows"
        assert radii.ndim == 1, "radii must be 1D array"
        assert (
            positions.shape[-1] == radii.shape[-1] + 1
        ), "radii must have n_nodes-1 elements"

        _radii = np.concatenate([radii, [0]])
        _radii[1:] += radii
        _radii[1:-1] /= 2.0
        for idx, sphere in enumerate(self.spheres):
            sphere.update_states(positions[:, idx], _radii[idx])

        for idx, cylinder in enumerate(self.cylinders):
            cylinder.update_states(
                positions[:, idx], positions[:, idx + 1], _radii[idx]
            )

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the material of the Rod object

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for the material update
        """
        for shperes in self.spheres:
            shperes.update_material(**kwargs)

        for cylinder in self.cylinders:
            cylinder.update_material(**kwargs)

    def update_keyframe(self, keyframe: int) -> None:
        """
        update keyframe for the rod object
        """
        for idx, sphere in enumerate(self.spheres):
            sphere.update_keyframe(keyframe)

        for idx, cylinder in enumerate(self.cylinders):
            cylinder.update_keyframe(keyframe)


class RodWithCylinder(RodWithSphereAndCylinder):
    """
    Rod class for managing visualization and rendering in Blender

    This class only creates cylinder objects

    Parameters
    ----------
    positions : NDArray
        The positions of the sphere objects. Expected shape is (n_dim, n_nodes).
        n_dim = 3
    radii : NDArray
        The radii of the sphere objects. Expected shape is (n_nodes-1,).

    """

    input_states = {"positions", "radii"}

    def __init__(self, positions: NDArray, radii: NDArray) -> None:
        # create cylinder objects
        self.cylinders: list[Cylinder] = []
        self._bpy_objs: dict[str, list[bpy.types.Object]] = {
            "cylinder": self.cylinders,
        }

        self.cylinders_material: list[bpy.types.Material] = []
        self._bpy_materials: dict[str, list[bpy.types.Material]] = {
            "cylinder": self.cylinders_material,
        }

        self._build(positions, radii)

    def _build(self, positions: NDArray, radii: NDArray) -> None:
        for j in range(radii.shape[-1]):
            cylinder = Cylinder(
                positions[:, j],
                positions[:, j + 1],
                radii[j],
            )
            self.cylinders.append(cylinder)
            self.cylinders_material.append(cylinder.material)

    def update_states(self, positions: NDArray, radii: NDArray) -> None:
        """
        Update the states of the rod object

        Parameters
        ----------
        positions : NDArray
            The positions of the sphere objects. Expected shape is (n_nodes, 3).
        radii : NDArray
            The radii of the sphere objects. Expected shape is (n_nodes-1,).
        """
        # check shape of positions and radii
        assert positions.ndim == 2, "positions must be 2D array"
        assert positions.shape[0] == 3, "positions must have 3 rows"
        assert radii.ndim == 1, "radii must be 1D array"
        assert (
            positions.shape[-1] == radii.shape[-1] + 1
        ), "radii must have n_nodes-1 elements"

        for idx, cylinder in enumerate(self.cylinders):
            cylinder.update_states(
                positions[:, idx], positions[:, idx + 1], radii[idx]
            )

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the material of the Rod object

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for the material update
        """
        for cylinder in self.cylinders:
            cylinder.update_material(**kwargs)

    def update_keyframe(self, keyframe: int) -> None:
        """
        Set keyframe for the rod object
        """
        for idx, cylinder in enumerate(self.cylinders):
            cylinder.update_keyframe(keyframe)


class RodWithBox(KeyFrameControlMixin):
    """
    Rod class for managing visualization and rendering in Blender

    This class creates sphere objects to represent position and cube to represent director

    Parameters
    ----------
    positions : NDArray
        The positions of the sphere objects. Expected shape is (n_dim, n_nodes).
        n_dim = 3
    radii : NDArray
        The radii of the sphere objects. Expected shape is (n_nodes-1,).

    """

    input_states = {"positions", "radii", "directors"}

    def __init__(
        self, positions: NDArray, radii: NDArray, directors: NDArray
    ) -> None:
        # create cylinder objects
        self.boxes: list[Box] = []
        self._bpy_objs: dict[str, list[bpy.types.Object]] = {
            "box": self.boxes,
        }
        self.boxes_material: list[bpy.types.Material] = []
        self._bpy_materials: dict[str, list[bpy.types.Material]] = {
            "box": self.boxes_material,
        }

        self._build(positions, radii, directors)

    @property
    def material(self) -> dict[str, list[bpy.types.Material]]:
        """
        Return the dictionary of Blender materials: sphere and cylinder
        """
        return self._bpy_materials

    @property
    def object(self) -> dict[str, list[bpy.types.Object]]:
        """
        Return the dictionary of Blender objects: sphere and cylinder
        """
        return self._bpy_objs

    @classmethod
    def create(cls, states: dict[str, NDArray]) -> "RodWithBox":
        """
        Basic factory method to create a new Rod object.
        States must have the following keys: positions(n_dim, n_nodes), radii(n_nodes-1,)

        Parameters
        ----------
        states: dict[str, NDArray]
            A dictionary where keys are state names and values are NDArrays.

        Returns
        -------
        RodWithSphereAndCylinder
            An object of Rod class containing the predefined states
        """
        positions = states["positions"]
        radii = states["radii"]
        directors = states["directors"]
        rod = cls(positions, radii, directors)
        return rod

    def _build(
        self, positions: NDArray, radii: NDArray, directors: NDArray
    ) -> None:
        n_elems = directors.shape[-1]
        for j in range(n_elems):
            box = Box(
                positions[:, j],
                positions[:, j + 1],
                radii[j],
                directors[..., j],
            )
            self.boxes.append(box)
            self.boxes_material.append(box.material)

    def update_states(
        self, positions: NDArray, radii: NDArray, directors: NDArray
    ) -> None:
        """
        Update the states of the rod object

        Parameters
        ----------
        positions : NDArray
            The positions of the sphere objects. Expected shape is (n_nodes, 3).
        radii : NDArray
            The radii of the sphere objects. Expected shape is (n_nodes-1,).
        """
        # check shape of positions and radii
        assert positions.ndim == 2, "positions must be 2D array"
        assert positions.shape[0] == 3, "positions must have 3 rows"
        assert radii.ndim == 1, "radii must be 1D array"
        assert (
            positions.shape[-1] == radii.shape[-1] + 1
        ), "radii must have n_nodes-1 elements"

        for idx, box in enumerate(self.boxes):
            box.update_states(
                positions[:, idx],
                positions[:, idx + 1],
                radii[idx],
                directors[..., idx],
            )

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the material of the Rod object

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for the material update
        """
        for obj in self.boxes:
            obj.update_material(**kwargs)

    def update_keyframe(self, keyframe: int) -> None:
        """
        Set keyframe for the rod object
        """
        for idx, box in enumerate(self.boxes):
            box.update_keyframe(keyframe)


class RodWithSpline(KeyFrameControlMixin):
    """
    Rod class for managing visualization and rendering in Blender using a Bezier spline pipe.

    Unlike RodWithSphereAndCylinder which uses discrete Sphere and Cylinder primitives,
    this class renders the rod as a single smooth BezierSplinePipe, producing a
    continuous surface that naturally interpolates between control points.

    Element-based radii (shape n_elems) are averaged at shared nodes before being
    passed to the spline, matching the convention of the other rod classes.

    Parameters
    ----------
    positions : NDArray
        The positions of the rod nodes. Expected shape is (n_dim, n_nodes), n_dim = 3.
    radii : NDArray
        The radius of each element. Expected shape is (n_elems,) = (n_nodes - 1,).
        Interior node radii are computed as the average of the two adjacent element radii.
    downsample_num_element : int or None, optional
        If provided, downsample the spline control points to this number.
        Must be at least 2. Default is None (no downsampling).
    """

    input_states = {"positions", "radii"}

    def __init__(
        self,
        positions: NDArray,
        radii: NDArray,
        downsample_num_element: int | None = None,
    ) -> None:
        node_radii = self._element_to_node_radii(radii)
        self._pipe = BezierSplinePipe(
            positions, node_radii, downsample_num_element
        )

    @staticmethod
    def _element_to_node_radii(radii: NDArray) -> NDArray:
        """
        Convert element-based radii (n_elems,) to node-based radii (n_nodes,).

        Node 0 and node n_elems keep the radius of their sole adjacent element.
        Interior nodes receive the average of the two adjacent element radii.
        """
        node_radii = np.concatenate([radii, [0.0]])
        node_radii[1:] += radii
        node_radii[1:-1] /= 2.0
        return node_radii

    @property
    def material(self) -> bpy.types.Material:
        """
        Return the Blender material of the spline pipe.
        """
        return self._pipe.material

    @property
    def object(self) -> bpy.types.Object:
        """
        Return the Blender object of the spline pipe.
        """
        return self._pipe.object

    @classmethod
    def create(cls, states: dict[str, NDArray]) -> "RodWithSpline":
        """
        Basic factory method to create a new RodWithSpline object.
        States must have the following keys: positions (3, n_nodes), radii (n_elems,).

        Parameters
        ----------
        states : dict[str, NDArray]
            A dictionary where keys are state names and values are NDArrays.

        Returns
        -------
        RodWithSpline
            An object containing the predefined states.
        """
        return cls(states["positions"], states["radii"])

    def update_states(self, positions: NDArray, radii: NDArray) -> None:
        """
        Update the states of the spline rod.

        Parameters
        ----------
        positions : NDArray
            Expected shape is (3, n_nodes).
        radii : NDArray
            The radius of each element. Expected shape is (n_elems,) = (n_nodes - 1,).
        """
        assert positions.ndim == 2, "positions must be 2D array"
        assert positions.shape[0] == 3, "positions must have 3 rows"
        assert radii.ndim == 1, "radii must be 1D array"
        assert (
            positions.shape[-1] == radii.shape[-1] + 1
        ), "radii must have n_nodes-1 elements"
        self._pipe.update_states(positions, self._element_to_node_radii(radii))

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the material of the spline rod.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for the material update (e.g. color as RGBA NDArray).
        """
        self._pipe.update_material(**kwargs)

    def update_keyframe(self, keyframe: int) -> None:
        """
        Set keyframe for the spline rod.
        """
        self._pipe.update_keyframe(keyframe)


class FinnedRodWithSpline(KeyFrameControlMixin):
    """
    Rod class rendering a hollow annulus pipe with two rectangular fins in Blender.

    Wraps ``BezierSplineFinnedPipe``.  Converts element-based arrays to node-based
    before delegating to the pipe primitive.

    The nominal cross-section geometry (``half_fin_span``, ``fin_thickness``,
    ``pipe_outer_radius``, ``inner_to_outer_radius_ratio``) and the fin offset
    distance (``r_rectangle_center``) are fixed at construction time.  Per-frame
    updates are driven by ``positions``, ``dilatation``, and ``fin_direction``.

    Parameters
    ----------
    positions : NDArray
        Node positions. Shape: (3, n_nodes).
    dilatation : NDArray
        Element stretch ratio. Shape: (n_elems,).
        Interior node values are averaged from adjacent elements.
    r_rectangle_center : NDArray
        Distance from the centerline to each fin's centre along d1, per element.
        Shape: (n_elems,).  Fixed at construction; not updated per frame.
    fin_direction : NDArray
        Element-based fin direction (d1). Shape: (3, n_elems).
    half_fin_span : float or NDArray
        Half-span (nominal width) of each rectangular fin cross-section. Default 1.0.
        When a scalar, the same span is applied to all elements.
        When a 1-D array of shape ``(n_elems,)``, each element gets its own span,
        enabling spatially varying fin geometry along the rod.
    fin_thickness : float
        Nominal thickness (depth) of each rectangular fin cross-section. Default 1.0.
    pipe_outer_radius : float or None, optional
        Outer radius of the annulus/circle centerline spine. Default None.
    inner_to_outer_radius_ratio : float or None, optional
        Ratio for hollow annulus inner radius. Default None (solid circle).
    downsample_num_element : int or None, optional
        Downsample spine control points. Default None.
    """

    input_states = {
        "positions",
        "dilatation",
        "r_rectangle_center",
        "fin_direction",
    }

    def __init__(
        self,
        positions: NDArray,
        dilatation: NDArray,
        r_rectangle_center: NDArray,
        fin_direction: NDArray,
        half_fin_span: float | NDArray = 1.0,
        fin_thickness: float = 1.0,
        pipe_outer_radius: float | None = None,
        inner_to_outer_radius_ratio: float | None = None,
        downsample_num_element: int | None = None,
    ) -> None:
        node_dilatation = self._element_to_node_values(dilatation)
        node_r = self._element_to_node_values(r_rectangle_center)
        node_d1 = self._element_to_node_fin_direction(fin_direction)
        self._pipe = BezierSplineFinnedPipe(
            positions,
            node_dilatation,
            node_r,
            node_d1,
            half_fin_span,
            fin_thickness,
            pipe_outer_radius,
            inner_to_outer_radius_ratio,
            downsample_num_element,
        )

    @staticmethod
    def _element_to_node_values(values: NDArray) -> NDArray:
        """
        Convert element-based values (n_elems,) to node-based values (n_nodes,).

        End nodes keep their sole adjacent element's value. Interior nodes
        receive the average of the two adjacent element values.
        """
        node_values = np.concatenate([values, [0.0]])
        node_values[1:] += values
        node_values[1:-1] /= 2.0
        return node_values

    @staticmethod
    def _element_to_node_fin_direction(d1: NDArray) -> NDArray:
        """
        Convert element-based fin direction (3, n_elems) to node-based (3, n_nodes).

        End nodes keep their sole adjacent element's fin direction. Interior nodes
        receive the normalised average of the two neighbouring element fin directions.
        """
        n_elems = d1.shape[-1]
        n_nodes = n_elems + 1
        node_d1 = np.empty((3, n_nodes))
        node_d1[:, 0] = d1[:, 0]
        node_d1[:, -1] = d1[:, -1]
        avg = (d1[:, :-1] + d1[:, 1:]) / 2.0
        norms = np.linalg.norm(avg, axis=0, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        node_d1[:, 1:-1] = avg / norms
        return node_d1

    @property
    def material(self) -> dict[str, bpy.types.Material]:
        """Return the material dict of the finned pipe."""
        return self._pipe.material

    @property
    def object(self) -> dict[str, bpy.types.Object]:
        """Return the object dict of the finned pipe."""
        return self._pipe.object

    @classmethod
    def create(
        cls,
        states: dict[str, NDArray],
        half_fin_span: float | NDArray = 1.0,
        fin_thickness: float = 1.0,
        pipe_outer_radius: float | None = None,
        inner_to_outer_radius_ratio: float | None = None,
        downsample_num_element: int | None = None,
    ) -> "FinnedRodWithSpline":
        """
        Factory method to create a new FinnedRodWithSpline.

        ``half_fin_span``, ``fin_thickness``, ``pipe_outer_radius``, and
        ``inner_to_outer_radius_ratio`` can be supplied as keyword arguments
        or inside ``states`` (scalar, 0-d array, or 1-D array of shape
        ``(n_elems,)``). Values found in ``states`` take precedence.

        Parameters
        ----------
        states : dict[str, NDArray]
            Must contain: ``positions`` (3, n_nodes), ``dilatation`` (n_elems,),
            ``r_rectangle_center`` (n_elems,), ``fin_direction`` (3, n_elems).
            May also contain ``half_fin_span`` as a scalar or per-element array,
            and ``fin_thickness``, ``pipe_outer_radius``,
            ``inner_to_outer_radius_ratio`` as per-rod scalars.
        """
        if "half_fin_span" in states:
            v = np.asarray(states["half_fin_span"])
            _half_fin_span: float | NDArray = v if v.ndim > 0 else float(v)
        else:
            _half_fin_span = half_fin_span
        _fin_thickness = (
            float(states["fin_thickness"])
            if "fin_thickness" in states
            else fin_thickness
        )
        _pipe_outer_radius = (
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
            states["r_rectangle_center"],
            states["fin_direction"],
            _half_fin_span,
            _fin_thickness,
            _pipe_outer_radius,
            _ratio,
            downsample_num_element,
        )

    def update_states(
        self,
        positions: NDArray,
        dilatation: NDArray,
        fin_direction: NDArray,
    ) -> None:
        """
        Update positions, dilatation, and fin_direction for the current frame.

        Parameters
        ----------
        positions : NDArray
            Shape: (3, n_nodes).
        dilatation : NDArray
            Element stretch ratio. Shape: (n_elems,).
        fin_direction : NDArray
            Element-based fin direction (d1). Shape: (3, n_elems).
        """
        assert positions.ndim == 2, "positions must be 2D array"
        assert positions.shape[0] == 3, "positions must have 3 rows"
        assert dilatation.ndim == 1, "dilatation must be 1D array"
        assert (
            positions.shape[-1] == dilatation.shape[-1] + 1
        ), "dilatation must have n_nodes-1 elements"
        self._pipe.update_states(
            positions,
            self._element_to_node_values(dilatation),
            self._element_to_node_fin_direction(fin_direction),
        )

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """Updates the material of the finned pipe."""
        self._pipe.update_material(**kwargs)

    def update_keyframe(self, keyframe: int) -> None:
        """Set keyframe for all spines in the finned pipe."""
        self._pipe.update_keyframe(keyframe)


class AnnulusRodWithSpline(KeyFrameControlMixin):
    """
    Rod class rendering a hollow annulus (or solid circle) pipe in Blender.

    Wraps ``BezierSplineAnnulusPipe``. Converts element-based ``dilatation``
    to node-based before delegating to the pipe primitive.

    The cross-section geometry (``pipe_outer_radius``, ``inner_to_outer_radius_ratio``)
    is fixed at construction. Per-frame updates are driven by ``positions``
    and ``dilatation``.

    Parameters
    ----------
    positions : NDArray
        Node positions. Shape: (3, n_nodes).
    dilatation : NDArray
        Element stretch ratio. Shape: (n_elems,).
        Interior node values are averaged from adjacent elements.
    pipe_outer_radius : float
        Outer radius of the annulus/circle bevel profile.
    inner_to_outer_radius_ratio : float or None, optional
        Ratio for hollow annulus inner radius. Default is None (solid circle).
    downsample_num_element : int or None, optional
        Downsample spine control points. Default is None.
    """

    input_states = {"positions", "dilatation"}

    def __init__(
        self,
        positions: NDArray,
        dilatation: NDArray,
        pipe_outer_radius: float,
        inner_to_outer_radius_ratio: float | None = None,
        downsample_num_element: int | None = None,
    ) -> None:
        node_dilatation = self._element_to_node_values(dilatation)
        self._pipe = BezierSplineAnnulusPipe(
            positions,
            node_dilatation,
            pipe_outer_radius,
            inner_to_outer_radius_ratio,
            downsample_num_element,
        )

    @staticmethod
    def _element_to_node_values(values: NDArray) -> NDArray:
        """Convert element-based (n_elems,) to node-based (n_nodes,)."""
        node_values = np.concatenate([values, [0.0]])
        node_values[1:] += values
        node_values[1:-1] /= 2.0
        return node_values

    @property
    def material(self) -> bpy.types.Material:
        """Return the Blender material of the annulus pipe."""
        return self._pipe.material

    @property
    def object(self) -> bpy.types.Object:
        """Return the Blender object of the annulus pipe."""
        return self._pipe.object

    @classmethod
    def create(
        cls,
        states: dict[str, NDArray],
        pipe_outer_radius: float = 1.0,
        inner_to_outer_radius_ratio: float | None = None,
        downsample_num_element: int | None = None,
    ) -> "AnnulusRodWithSpline":
        """
        Factory method to create a new AnnulusRodWithSpline.

        ``pipe_outer_radius`` and ``inner_to_outer_radius_ratio`` can be
        supplied as keyword arguments or inside ``states`` (scalar or 0-d
        array). Values found in ``states`` take precedence.

        Parameters
        ----------
        states : dict[str, NDArray]
            Must contain: ``positions`` (3, n_nodes), ``dilatation`` (n_elems,).
            May also contain ``pipe_outer_radius`` and
            ``inner_to_outer_radius_ratio``.
        """
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

    def update_states(self, positions: NDArray, dilatation: NDArray) -> None:
        """
        Update positions and dilatation for the current frame.

        Parameters
        ----------
        positions : NDArray
            Shape: (3, n_nodes).
        dilatation : NDArray
            Element stretch ratio. Shape: (n_elems,).
        """
        assert positions.ndim == 2, "positions must be 2D array"
        assert positions.shape[0] == 3, "positions must have 3 rows"
        assert dilatation.ndim == 1, "dilatation must be 1D array"
        assert (
            positions.shape[-1] == dilatation.shape[-1] + 1
        ), "dilatation must have n_nodes-1 elements"
        self._pipe.update_states(
            positions,
            self._element_to_node_values(dilatation),
        )

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """Update the material of the annulus pipe."""
        self._pipe.update_material(**kwargs)

    def update_keyframe(self, keyframe: int) -> None:
        """Set keyframe for the annulus pipe."""
        self._pipe.update_keyframe(keyframe)


class RectAnnulusRodWithSpline(KeyFrameControlMixin):
    """
    Rod class rendering a rectangular-tube-with-circular-bore pipe in Blender.

    Wraps ``BezierSplineRectAnnulusPipe``. Converts element-based ``dilatation``
    to node-based before delegating to the pipe primitive.

    The cross-section geometry (``rect_width``, ``rect_depth``, ``bore_radius``)
    is fixed at construction. Per-frame updates are driven by ``positions``
    and ``dilatation``.

    Parameters
    ----------
    positions : NDArray
        Node positions. Shape: (3, n_nodes).
    dilatation : NDArray
        Element stretch ratio. Shape: (n_elems,).
        Interior node values are averaged from adjacent elements.
    rect_width : float
        Full width of the outer rectangle cross-section.
    rect_depth : float
        Full depth of the outer rectangle cross-section.
    bore_radius : float
        Radius of the circular bore through the center.
        Must be < min(rect_width, rect_depth) / 2.
    downsample_num_element : int or None, optional
        Downsample spine control points. Default is None.
    """

    input_states = {"positions", "dilatation"}

    def __init__(
        self,
        positions: NDArray,
        dilatation: NDArray,
        rect_width: float,
        rect_depth: float,
        bore_radius: float,
        downsample_num_element: int | None = None,
    ) -> None:
        node_dilatation = self._element_to_node_values(dilatation)
        self._pipe = BezierSplineRectAnnulusPipe(
            positions,
            node_dilatation,
            rect_width,
            rect_depth,
            bore_radius,
            downsample_num_element,
        )

    @staticmethod
    def _element_to_node_values(values: NDArray) -> NDArray:
        """Convert element-based (n_elems,) to node-based (n_nodes,)."""
        node_values = np.concatenate([values, [0.0]])
        node_values[1:] += values
        node_values[1:-1] /= 2.0
        return node_values

    @property
    def material(self) -> bpy.types.Material:
        """Return the Blender material of the rect-annulus pipe."""
        return self._pipe.material

    @property
    def object(self) -> bpy.types.Object:
        """Return the Blender object of the rect-annulus pipe."""
        return self._pipe.object

    @classmethod
    def create(
        cls,
        states: dict[str, NDArray],
        rect_width: float = 1.0,
        rect_depth: float = 1.0,
        bore_radius: float = 0.25,
        downsample_num_element: int | None = None,
    ) -> "RectAnnulusRodWithSpline":
        """
        Factory method to create a new RectAnnulusRodWithSpline.

        ``rect_width``, ``rect_depth``, and ``bore_radius`` can be supplied as
        keyword arguments or inside ``states`` (scalar or 0-d array). Values
        found in ``states`` take precedence.

        Parameters
        ----------
        states : dict[str, NDArray]
            Must contain: ``positions`` (3, n_nodes), ``dilatation`` (n_elems,).
            May also contain ``rect_width``, ``rect_depth``, and ``bore_radius``
            as per-rod scalars.
        """
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

    def update_states(self, positions: NDArray, dilatation: NDArray) -> None:
        """
        Update positions and dilatation for the current frame.

        Parameters
        ----------
        positions : NDArray
            Shape: (3, n_nodes).
        dilatation : NDArray
            Element stretch ratio. Shape: (n_elems,).
        """
        assert positions.ndim == 2, "positions must be 2D array"
        assert positions.shape[0] == 3, "positions must have 3 rows"
        assert dilatation.ndim == 1, "dilatation must be 1D array"
        assert (
            positions.shape[-1] == dilatation.shape[-1] + 1
        ), "dilatation must have n_nodes-1 elements"
        self._pipe.update_states(
            positions,
            self._element_to_node_values(dilatation),
        )

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """Update the material of the rect-annulus pipe."""
        self._pipe.update_material(**kwargs)

    def update_keyframe(self, keyframe: int) -> None:
        """Set keyframe for the rect-annulus pipe."""
        self._pipe.update_keyframe(keyframe)


# Alias
Rod = RodWithSphereAndCylinder

if TYPE_CHECKING:
    data = {
        "positions": np.array([[0, 0, 0], [1, 1, 1]]),
        "radii": np.array([1.0, 1.0]),
    }
    _: CompositeProtocol = RodWithSphereAndCylinder.create(data)
