# -*- coding: utf-8 -*-

from typing import Tuple

from fury import window

from tractolearn.visualization.utils import AnatomicalView


def transform_scene_focal_point(scene, anatomical_view=AnatomicalView.AXIAL_SUPERIOR):

    # Re-initialize camera
    position = [0, 0, 1]
    focal_point = [0, 0, 0]
    view_up = [0, 1, 0]
    scene.set_camera(position, focal_point, view_up)

    # ToDo
    # Assuming that the volume in the scene is always oriented the same way?
    if anatomical_view == AnatomicalView.AXIAL_SUPERIOR:
        pass
    elif anatomical_view == AnatomicalView.AXIAL_INFERIOR:
        scene.pitch(180)
    elif anatomical_view == AnatomicalView.CORONAL_ANTERIOR:
        scene.pitch(270)
        scene.set_camera(view_up=(0, 0, 1))
    elif anatomical_view == AnatomicalView.CORONAL_POSTERIOR:
        scene.pitch(90)
        scene.set_camera(view_up=(0, 0, 1))
    elif anatomical_view == AnatomicalView.SAGITTAL_LEFT:
        scene.yaw(-90)
        scene.roll(90)
    elif anatomical_view == AnatomicalView.SAGITTAL_RIGHT:
        scene.yaw(90)
        scene.roll(-90)
    else:
        raise ValueError(
            "Unknown anatomical view type.\n"
            "Found: {}; Available: {}".format(
                anatomical_view, AnatomicalView._member_names_
            )
        )

    scene.reset_camera()


def compose_scene(
    actors,
    anatomical_view=AnatomicalView.AXIAL_SUPERIOR,
    background=window.colors.black,
):

    scene = window.Scene()

    [scene.add(_actor) for _actor in actors]

    scene.background(background)

    transform_scene_focal_point(scene, anatomical_view)

    return scene
