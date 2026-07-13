import os
import sys
from pathlib import Path

import click
import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover

    def tqdm(iterable, **kwargs):
        return iterable


import bsr


def confirm_pyelastica_npz_structure(
    path: str,
    tags: list[str] | None = None,
    finned_spline_rod: bool = False,
    annulus_rod: bool = False,
    rect_annulus_rod: bool = False,
    fin_pipe_bundle: bool = False,
) -> None:
    data = np.load(path)
    keys = list(data.keys())

    # Check if all keys match one of the following patterns
    required_key_pattern = ["time"]
    if fin_pipe_bundle:
        prefixes: list[str | None] = (
            list(tags) if tags is not None else [None]
        )
        for tag in prefixes:
            prefix = f"{tag}_" if tag is not None else ""
            required_key_pattern.append(prefix + "fin_position_history")
            required_key_pattern.append(prefix + "fin_dilatation_history")
            required_key_pattern.append(prefix + "fin_left_span")
            required_key_pattern.append(prefix + "fin_right_span")
            required_key_pattern.append(prefix + "fin_thickness")
            required_key_pattern.append(prefix + "pipe_position_history")
            required_key_pattern.append(prefix + "pipe_dilatation_history")
            required_key_pattern.append(prefix + "pipe_outer_radius")
            required_key_pattern.append(
                prefix + "pipe_inner_to_outer_radius_ratio"
            )
    elif tags is not None:
        for tag in tags:
            required_key_pattern.append(tag + "_position_history")
            if finned_spline_rod:
                required_key_pattern.append(tag + "_dilatation_history")
                required_key_pattern.append(tag + "_fin_direction_history")
                required_key_pattern.append(tag + "_r_rectangle_center")
                required_key_pattern.append(tag + "_half_fin_span")
                required_key_pattern.append(tag + "_fin_thickness")
                required_key_pattern.append(tag + "_pipe_outer_radius")
                required_key_pattern.append(
                    tag + "_inner_to_outer_radius_ratio"
                )
            elif annulus_rod:
                required_key_pattern.append(tag + "_dilatation_history")
                required_key_pattern.append(tag + "_pipe_outer_radius")
                required_key_pattern.append(
                    tag + "_inner_to_outer_radius_ratio"
                )
            elif rect_annulus_rod:
                required_key_pattern.append(tag + "_dilatation_history")
                required_key_pattern.append(tag + "_rect_width")
                required_key_pattern.append(tag + "_rect_depth")
                required_key_pattern.append(tag + "_bore_radius")
            else:
                required_key_pattern.append(tag + "_radius_history")
    else:
        required_key_pattern.append("position_history")
        if finned_spline_rod:
            required_key_pattern.append("dilatation_history")
            required_key_pattern.append("fin_direction_history")
            required_key_pattern.append("r_rectangle_center")
            required_key_pattern.append("half_fin_span")
            required_key_pattern.append("fin_thickness")
            required_key_pattern.append("pipe_outer_radius")
            required_key_pattern.append("inner_to_outer_radius_ratio")
        elif annulus_rod:
            required_key_pattern.append("dilatation_history")
            required_key_pattern.append("pipe_outer_radius")
            required_key_pattern.append("inner_to_outer_radius_ratio")
        elif rect_annulus_rod:
            required_key_pattern.append("dilatation_history")
            required_key_pattern.append("rect_width")
            required_key_pattern.append("rect_depth")
            required_key_pattern.append("bore_radius")
        else:
            required_key_pattern.append("radius_history")

    for key in keys:
        if not any([pattern in key for pattern in required_key_pattern]):
            raise KeyError(
                f"Key {key} does not match any of the required patterns."
            )


def construct_blender_file(
    path: str | Path,
    output: str | Path,
    tags: list[str] | None,
    fps: int = -1,
    spline_rod: bool = False,
    finned_spline_rod: bool = False,
    annulus_rod: bool = False,
    rect_annulus_rod: bool = False,
    fin_pipe_bundle: bool = False,
) -> None:
    """
    Read npz file containing the position and radius data of multiple elastica rods.
    The shape of time is [n_timesteps]
    The shape of position is [n_rods, n_timesteps, 3, n_nodes]
    The shape of radius is [n_rods, n_timesteps, n_elems]

    When spline_rod is True, SplineRodStack is used via create_spline_rod_collection.
    Each rod is rendered as a single smooth BezierSplinePipe. The NPZ file requires
    only position_history and radius_history (or <tag>_*_history variants), same as
    the default case.

    When finned_spline_rod is True, both create_spline_rod_collection and
    create_spline_finned_rod_collection are created and overlaid on the same
    position history. The circular spline renders the rod centerline while the
    rectangular spline renders the fin cross-section scaled by dilatation. A Bezier
    circle can also be embedded in the rectangular bevel profile. The NPZ file must
    contain position_history, radius_history, dilatation_history, half_fin_span,
    fin_thickness, and pipe_outer_radius
    (or <tag>_*_history / <tag>_half_fin_span / <tag>_fin_thickness /
    <tag>_pipe_outer_radius variants). ``half_fin_span``, ``fin_thickness``, and
    ``pipe_outer_radius`` are per-rod scalar arrays (shape n_rods) baked into the
    bevel profile at construction; not updated per frame.

    When fin_pipe_bundle is True, create_fin_segment_rod_collection and
    create_annulus_rod_collection are built from independent ``fin_*`` and
    ``pipe_*`` key groups (no shared centerline position history -- each fin
    segment and each pipe already carries its own fully-resolved spine).
    Requires fin_position_history, fin_dilatation_history, fin_left_span,
    fin_right_span, fin_thickness, pipe_position_history,
    pipe_dilatation_history, pipe_outer_radius, and
    pipe_inner_to_outer_radius_ratio (or <tag>_fin_*/<tag>_pipe_* variants).
    ``fin_left_span``/``fin_right_span`` are per-fin, per-element arrays;
    ``fin_thickness``, ``pipe_outer_radius``, and
    ``pipe_inner_to_outer_radius_ratio`` are per-rod scalar arrays baked in
    at construction; none of these are updated per frame.

    Parameters
    ----------
    path : str or Path
        Path to the input NPZ file.
    output : str or Path
        Path to the output .blend file.
    tags : list[str] or None
        Tags to identify individual rods in the NPZ file. If None, uses
        unprefixed keys (e.g. position_history).
    fps : int, optional
        Frames per second. Default is -1 (every timestep saved as one frame).
    spline_rod : bool, optional
        Use SplineRodStack (create_spline_rod_collection) to render each rod as a
        smooth BezierSplinePipe. Requires only position_history and radius_history,
        same as the default. Default is False.
    finned_spline_rod : bool, optional
        Overlay create_spline_rod_collection and create_spline_finned_rod_collection
        on the same position history. Requires position_history, radius_history,
        dilatation_history, half_fin_span, fin_thickness, and pipe_outer_radius
        (per-rod scalars, shape n_rods). No director_history needed. Default is False.
    fin_pipe_bundle : bool, optional
        Build independent fin-segment and off-center pipe collections from
        ``fin_*``/``pipe_*`` key groups. See above. Default is False.
    """
    bsr.clear_mesh_objects()
    confirm_pyelastica_npz_structure(
        str(path),
        tags,
        finned_spline_rod=finned_spline_rod,
        annulus_rod=annulus_rod,
        rect_annulus_rod=rect_annulus_rod,
        fin_pipe_bundle=fin_pipe_bundle,
    )
    data = np.load(path)

    time = data["time"]
    start_time, end_time = time[0], time[-1]
    if fps == -1:
        probe_time = np.arange(time.size)
    else:  # pragma: no cover
        NotImplementedError("Not implemented yet.")

    if fin_pipe_bundle:
        prefixes: list[str | None] = (
            list(tags) if tags is not None else [None]
        )
        for tag in prefixes:
            prefix = f"{tag}_" if tag is not None else ""
            fin_position_history = data[prefix + "fin_position_history"]
            fin_dilatation_history = data[prefix + "fin_dilatation_history"]
            fin_left_span = data[prefix + "fin_left_span"]
            fin_right_span = data[prefix + "fin_right_span"]
            fin_thickness = data[prefix + "fin_thickness"]
            fin_init_state = {
                "positions": fin_position_history[:, 0, ...],
                "dilatation": fin_dilatation_history[:, 0, ...],
                "left_span": fin_left_span,
                "right_span": fin_right_span,
                "fin_thickness": fin_thickness,
            }
            fin_rods = bsr.create_fin_segment_rod_collection(fin_init_state)

            pipe_position_history = data[prefix + "pipe_position_history"]
            pipe_dilatation_history = data[
                prefix + "pipe_dilatation_history"
            ]
            pipe_outer_radius = data[prefix + "pipe_outer_radius"]
            pipe_inner_to_outer_radius_ratio = data[
                prefix + "pipe_inner_to_outer_radius_ratio"
            ]
            pipe_init_state = {
                "positions": pipe_position_history[:, 0, ...],
                "dilatation": pipe_dilatation_history[:, 0, ...],
                "pipe_outer_radius": pipe_outer_radius,
                "inner_to_outer_radius_ratio": pipe_inner_to_outer_radius_ratio,
            }
            pipe_rods = bsr.create_annulus_rod_collection(pipe_init_state)

            for tidx, _ in tqdm(enumerate(time), total=len(time)):
                fin_rods.update_states(
                    fin_position_history[:, tidx, ...],
                    fin_dilatation_history[:, tidx, ...],
                )
                fin_rods.update_keyframe(tidx)
                pipe_rods.update_states(
                    pipe_position_history[:, tidx, ...],
                    pipe_dilatation_history[:, tidx, ...],
                )
                pipe_rods.update_keyframe(tidx)
    elif tags is None:
        position_history = data["position_history"]
        if finned_spline_rod:
            dilatation_history = data["dilatation_history"]
            fin_direction_history = data["fin_direction_history"]
            r_rectangle_center = data["r_rectangle_center"]
            half_fin_span = data["half_fin_span"]
            fin_thickness = data["fin_thickness"]
            pipe_outer_radius = data["pipe_outer_radius"]
            inner_to_outer_radius_ratio = data["inner_to_outer_radius_ratio"]
            rect_init_state = {
                "positions": position_history[:, 0, ...],
                "dilatation": dilatation_history[:, 0, ...],
                "fin_direction": fin_direction_history[:, 0, ...],
                "r_rectangle_center": r_rectangle_center,
                "half_fin_span": half_fin_span,
                "fin_thickness": fin_thickness,
                "pipe_outer_radius": pipe_outer_radius,
                "inner_to_outer_radius_ratio": inner_to_outer_radius_ratio,
            }
            rect_rods = bsr.create_spline_finned_rod_collection(rect_init_state)
            for tidx, _ in tqdm(enumerate(time), total=len(time)):
                rect_rods.update_states(
                    position_history[:, tidx, ...],
                    dilatation_history[:, tidx, ...],
                    fin_direction_history[:, tidx, ...],
                )
                rect_rods.update_keyframe(tidx)
        elif annulus_rod:
            dilatation_history = data["dilatation_history"]
            pipe_outer_radius = data["pipe_outer_radius"]
            inner_to_outer_radius_ratio = data["inner_to_outer_radius_ratio"]
            annulus_init_state = {
                "positions": position_history[:, 0, ...],
                "dilatation": dilatation_history[:, 0, ...],
                "pipe_outer_radius": pipe_outer_radius,
                "inner_to_outer_radius_ratio": inner_to_outer_radius_ratio,
            }
            annulus_rods = bsr.create_annulus_rod_collection(annulus_init_state)
            for tidx, _ in tqdm(enumerate(time), total=len(time)):
                annulus_rods.update_states(
                    position_history[:, tidx, ...],
                    dilatation_history[:, tidx, ...],
                )
                annulus_rods.update_keyframe(tidx)
        elif rect_annulus_rod:
            dilatation_history = data["dilatation_history"]
            rect_width = data["rect_width"]
            rect_depth = data["rect_depth"]
            bore_radius = data["bore_radius"]
            rect_annulus_init_state = {
                "positions": position_history[:, 0, ...],
                "dilatation": dilatation_history[:, 0, ...],
                "rect_width": rect_width,
                "rect_depth": rect_depth,
                "bore_radius": bore_radius,
            }
            rect_annulus_rods = bsr.create_rect_annulus_rod_collection(
                rect_annulus_init_state
            )
            for tidx, _ in tqdm(enumerate(time), total=len(time)):
                rect_annulus_rods.update_states(
                    position_history[:, tidx, ...],
                    dilatation_history[:, tidx, ...],
                )
                rect_annulus_rods.update_keyframe(tidx)
        elif spline_rod:
            radius_history = data["radius_history"]
            init_state = {
                "positions": position_history[:, 0, ...],
                "radii": radius_history[:, 0, ...],
            }
            rods = bsr.create_spline_rod_collection(init_state)
            for tidx, _ in tqdm(enumerate(time), total=len(time)):
                rods.update_states(
                    position_history[:, tidx, ...], radius_history[:, tidx, ...]
                )
                rods.update_keyframe(tidx)
        else:
            radius_history = data["radius_history"]
            init_state = {
                "positions": position_history[:, 0, ...],
                "radii": radius_history[:, 0, ...],
            }
            rods = bsr.create_rod_collection(init_state)
            for tidx, _ in tqdm(enumerate(time), total=len(time)):
                rods.update_states(
                    position_history[:, tidx, ...], radius_history[:, tidx, ...]
                )
                rods.update_keyframe(tidx)
    else:
        for tag in tags:
            position_history = data[tag + "_position_history"]
            if finned_spline_rod:
                dilatation_history = data[tag + "_dilatation_history"]
                fin_direction_history = data[tag + "_fin_direction_history"]
                r_rectangle_center = data[tag + "_r_rectangle_center"]
                half_fin_span = data[tag + "_half_fin_span"]
                fin_thickness = data[tag + "_fin_thickness"]
                pipe_outer_radius = data[tag + "_pipe_outer_radius"]
                inner_to_outer_radius_ratio = data[
                    tag + "_inner_to_outer_radius_ratio"
                ]
                rect_init_state = {
                    "positions": position_history[:, 0, ...],
                    "dilatation": dilatation_history[:, 0, ...],
                    "fin_direction": fin_direction_history[:, 0, ...],
                    "r_rectangle_center": r_rectangle_center,
                    "half_fin_span": half_fin_span,
                    "fin_thickness": fin_thickness,
                    "pipe_outer_radius": pipe_outer_radius,
                    "inner_to_outer_radius_ratio": inner_to_outer_radius_ratio,
                }
                rect_rods = bsr.create_spline_finned_rod_collection(
                    rect_init_state
                )
                for tidx, _ in tqdm(enumerate(time), total=len(time)):
                    rect_rods.update_states(
                        position_history[:, tidx, ...],
                        dilatation_history[:, tidx, ...],
                        fin_direction_history[:, tidx, ...],
                    )
                    rect_rods.update_keyframe(tidx)
            elif annulus_rod:
                dilatation_history = data[tag + "_dilatation_history"]
                pipe_outer_radius = data[tag + "_pipe_outer_radius"]
                inner_to_outer_radius_ratio = data[
                    tag + "_inner_to_outer_radius_ratio"
                ]
                annulus_init_state = {
                    "positions": position_history[:, 0, ...],
                    "dilatation": dilatation_history[:, 0, ...],
                    "pipe_outer_radius": pipe_outer_radius,
                    "inner_to_outer_radius_ratio": inner_to_outer_radius_ratio,
                }
                annulus_rods = bsr.create_annulus_rod_collection(
                    annulus_init_state
                )
                for tidx, _ in tqdm(enumerate(time), total=len(time)):
                    annulus_rods.update_states(
                        position_history[:, tidx, ...],
                        dilatation_history[:, tidx, ...],
                    )
                    annulus_rods.update_keyframe(tidx)
            elif rect_annulus_rod:
                dilatation_history = data[tag + "_dilatation_history"]
                rect_width = data[tag + "_rect_width"]
                rect_depth = data[tag + "_rect_depth"]
                bore_radius = data[tag + "_bore_radius"]
                rect_annulus_init_state = {
                    "positions": position_history[:, 0, ...],
                    "dilatation": dilatation_history[:, 0, ...],
                    "rect_width": rect_width,
                    "rect_depth": rect_depth,
                    "bore_radius": bore_radius,
                }
                rect_annulus_rods = bsr.create_rect_annulus_rod_collection(
                    rect_annulus_init_state
                )
                for tidx, _ in tqdm(enumerate(time), total=len(time)):
                    rect_annulus_rods.update_states(
                        position_history[:, tidx, ...],
                        dilatation_history[:, tidx, ...],
                    )
                    rect_annulus_rods.update_keyframe(tidx)
            elif spline_rod:
                radius_history = data[tag + "_radius_history"]
                init_state = {
                    "positions": position_history[:, 0, ...],
                    "radii": radius_history[:, 0, ...],
                }
                rods = bsr.create_spline_rod_collection(init_state)
                for tidx, _ in tqdm(enumerate(time), total=len(time)):
                    rods.update_states(
                        position_history[:, tidx, ...],
                        radius_history[:, tidx, ...],
                    )
                    rods.update_keyframe(tidx)
            else:
                radius_history = data[tag + "_radius_history"]
                init_state = {
                    "positions": position_history[:, 0, ...],
                    "radii": radius_history[:, 0, ...],
                }
                rods = bsr.create_rod_collection(init_state)
                for tidx, _ in tqdm(enumerate(time), total=len(time)):
                    rods.update_states(
                        position_history[:, tidx, ...],
                        radius_history[:, tidx, ...],
                    )
                    rods.update_keyframe(tidx)

    bsr.save(output)


@click.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    help="Path to the npz file containing the data.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    help="Path to the output blend file.",
)
@click.option(
    "--tags",
    "-t",
    type=str,
    multiple=True,
    default=None,
    help="Tags to identify rods.",
)
@click.option(
    "--fps",
    type=int,
    default=-1,
    help="Frames per second. By default, the value is set to -1, which means every frame is saved.",
)
@click.option(
    "--spline-rod",
    is_flag=True,
    default=False,
    help="Use SplineRodStack to render each rod as a smooth BezierSplinePipe. "
    "Requires only position_history and radius_history in the NPZ file.",
)
@click.option(
    "--finned-spline-rod",
    is_flag=True,
    default=False,
    help="Overlay create_spline_rod_collection and create_spline_finned_rod_collection "
    "on the same position history. Requires position_history, radius_history, "
    "dilatation_history, width, and depth in the NPZ file.",
)
@click.option(
    "--annulus-rod",
    is_flag=True,
    default=False,
    help="Use AnnulusRodStack to render each rod as a hollow annulus (or solid circle) pipe. "
    "Requires position_history, dilatation_history, pipe_outer_radius, and "
    "inner_to_outer_radius_ratio in the NPZ file.",
)
@click.option(
    "--rect-annulus-rod",
    is_flag=True,
    default=False,
    help="Use RectAnnulusRodStack to render each rod as a rectangular tube with a circular bore. "
    "Requires position_history, dilatation_history, rect_width, rect_depth, and "
    "bore_radius in the NPZ file.",
)
@click.option(
    "--fin-pipe-bundle",
    is_flag=True,
    default=False,
    help="Build independent fin-segment (create_fin_segment_rod_collection) and "
    "off-center pipe (create_annulus_rod_collection) collections from fin_* and "
    "pipe_* key groups. Requires fin_position_history, fin_dilatation_history, "
    "fin_left_span, fin_right_span, fin_thickness, pipe_position_history, "
    "pipe_dilatation_history, pipe_outer_radius, and "
    "pipe_inner_to_outer_radius_ratio in the NPZ file.",
)
def main(
    path: Path,
    output: Path,
    tags: list[str],
    fps: int,
    spline_rod: bool,
    finned_spline_rod: bool,
    annulus_rod: bool,
    rect_annulus_rod: bool,
    fin_pipe_bundle: bool,
) -> None:  # pragma: no cover
    construct_blender_file(
        path,
        output,
        tags,
        fps,
        spline_rod,
        finned_spline_rod,
        annulus_rod,
        rect_annulus_rod,
        fin_pipe_bundle,
    )
