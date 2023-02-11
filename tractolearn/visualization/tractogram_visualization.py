# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np
from fury import actor, window
from fury.colormap import line_colors
from nibabel.nifti1 import Nifti1Image
from nibabel.streamlines.array_sequence import ArraySequence

from tractolearn.visualization.scene_utils import compose_scene
from tractolearn.visualization.utils import (
    transform_streamlines_according_to_target,
)


# ToDo
# Add the possibility to add the reference anatomy and peaks actors to the
# plots if desired.
# ToDo/Fixme
# The naming may be misleading since the method takes an ArraySequence as
# it input instead of a tractogram.
# This should be made consistent at some point.
def plot_tractogram(
    streamlines,
    anat_ref_img,
    show_anat_ref=False,
    show_endpoints=False,
    show_endpoints_labels=False,
    show_origin=False,
    colormap=None,
    scale_range=None,
    hue: Tuple[float, float] = (0.8, 0),
    saturation: Tuple[float, float] = (1.0, 1.0),
    size: Tuple[int, int] = (600, 600),
    interactive=True,
    show_bar=False,
    filename=None,
    axis: int = 0,
) -> None:
    """Plots a tractogram and saves the scene to a PNG file if a filename is
    provided. The streamlines' endpoints (heads and tails) may be shown with a
    glyph. The head of a streamline is assumed to be its first index, whereas
    its last index is assumed to be its tail. The glyph of the head is
    displayed in white color, being red for the tail. Labels for the first
    streamline's endpoints may be displayed for reference. The glyph of the
    origin is displayed in yellow color if it is to be displayed. If no
    colormap is provided, a default one is generated from the streamlines.

    Parameters
    ----------
    streamlines : nib.streamlines.ArraySequence
        Streamlines to be plot.
    anat_ref_img : nib.nifti1.Nifti1Image
        Anatomical reference image.
    show_anat_ref : bool, optional
        True if the anatomical reference image is to be displayed.
    show_endpoints : bool, optional
        True if the streamlines' endpoints are to be displayed with a glyph.
    show_endpoints_labels : bool, optional
        True if the first streamlines endpoints' labels are to be displayed.
    show_origin : bool, optional
        True if the origin (0, 0, 0) is to be displayed with a glyph.
    colormap : ndarray, optional
        Colormap.
    scale_range : tuple
        Min and max values of the colormap scale to be shown.
    hue : Tuple[float, float], optional
        Colormap hue.
    saturation : Tuple[float, float], optional
        Colormap saturation.
    size : Tuple[int, int], optional
        Size of the render window.
    interactive : bool, optional
        True if the scene is to be rendered.
    show_bar : bool, optional
        True if the color bar is to be shown.
    filename : str, optional
        Filename to save the scene.
    """

    actors = []

    # If no colormap is provided, generate a colormap from the streamlines
    if colormap is None:
        colormap = line_colors(streamlines)

    # If no colormap is provided, generate a colormap from the colormap
    if scale_range is None:
        scale_range = (colormap.min(), colormap.max())

    lut_cmap = actor.colormap_lookup_table(
        scale_range=scale_range, hue_range=hue, saturation_range=saturation
    )

    stream_actor = actor.line(
        streamlines, colormap, linewidth=1.0, lookup_colormap=lut_cmap
    )
    actors.append(stream_actor)

    if show_anat_ref:
        anat_ref_img_data = anat_ref_img.get_fdata()
        slice_actor = actor.slicer(anat_ref_img_data, opacity=0.5)
        if axis == 2:
            slice_actor.display(None, None, slice_actor.shape[2] // 2)
        elif axis == 1:
            slice_actor.display(None, slice_actor.shape[1] // 2, None)
        elif axis == 0:
            slice_actor.display(slice_actor.shape[0] // 2, None, None)
        else:
            raise ValueError("Axis must 0, 1 or 2, " f"Got {axis}.")
        actors.append(slice_actor)

    if show_endpoints:
        heads = streamlines.get_data()[streamlines._offsets]
        tails_indices = np.cumsum(streamlines._lengths) - 1
        tails = streamlines.get_data()[tails_indices]

        head_color = (1.0, 1.0, 1.0)
        head_actor = actor.dots(heads, color=head_color)
        tail_color = (1.0, 0.0, 0.0)
        tail_actor = actor.dots(tails, color=tail_color)

        actors.append(head_actor)
        actors.append(tail_actor)

        if show_endpoints_labels:
            head_label_text = "H"
            label_scale = (3.0, 3.0, 3.0)
            head_label_actor = actor.label(
                head_label_text,
                pos=heads[0],
                scale=label_scale,
                color=head_color,
            )
            tail_label_text = "T"
            tail_label_actor = actor.label(
                tail_label_text,
                pos=tails[0],
                scale=label_scale,
                color=tail_color,
            )

            actors.append(head_label_actor)
            actors.append(tail_label_actor)

    if show_origin:
        center_color = (1.0, 1.0, 0.0)
        center_actor = actor.dots(np.array([0, 0, 0]), color=center_color)
        actors.append(center_actor)

    if show_bar:
        bar = actor.scalar_bar(lut_cmap)
        actors.append(bar)

    scene = compose_scene(actors)

    if interactive:
        window.show(scene, size=size, reset_camera=False)

    if filename is not None:
        window.snapshot(scene, fname=filename, size=size)

    scene.clear()


def plot_tractogram_with_anat_ref(
    streamlines: ArraySequence,
    anat_ref_img: Nifti1Image,
    filename: str = None,
    show_anat_ref: bool = True,
    show_endpoints: bool = False,
    colormap: np.array = None,
    scale_range: Tuple[float, float] = None,
    size: Tuple[int, int] = (1920, 1080),
    interactive: bool = False,
    show_bar: bool = False,
    axis: int = 0,
) -> None:

    transformed_streamlines = transform_streamlines_according_to_target(
        streamlines, anat_ref_img, anat_ref_img.affine
    )

    plot_tractogram(
        transformed_streamlines,
        anat_ref_img,
        show_anat_ref=show_anat_ref,
        show_endpoints=show_endpoints,
        colormap=colormap,
        scale_range=scale_range,
        size=size,
        interactive=interactive,
        show_bar=show_bar,
        filename=filename,
        axis=axis,
    )
