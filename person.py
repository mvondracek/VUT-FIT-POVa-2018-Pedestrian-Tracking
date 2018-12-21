"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import datetime
from typing import List

import numpy as np

from camera import Camera


class PersonView:
    """
    Single view of a person detected in an image from given camera.
    View includes detected position of person's pose.
    """

    def __init__(self, person_image, camera: Camera, pose_top_coordinate, pose_bottom_coordinate):
        self.person_image = person_image
        self.camera = camera
        self.pose_top_coordinate = pose_top_coordinate
        self.pose_bottom_coordinate = pose_bottom_coordinate


class PersonTimeFrame:
    """
    Time frame when given person was detected on at least one view of the scene.
    """

    def __init__(self, views: List[PersonView], time=datetime.datetime.now()):
        self.time = time
        self.views = views
        self.coordinates_3d = None

        self.distance_planes = []  # TODO This is used only for visualization
        self.real_subject_coordinates_3d = None  # TODO This is used only for testing
        self.intersection_line = None  # TODO This is used only for visualization


def plot_person_time_frame(person_time_frame: PersonTimeFrame):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # required for `ax = fig.add_subplot(111, projection='3d')`
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # requires `from mpl_toolkits.mplot3d import Axes3D`
    plot_size_x = 500
    plot_size_y = 500
    plot_size_z = 30
    distance_plane_size_xy = 100
    ray_extender = 100
    intersection_line_extender = 100

    # walls
    ax.plot([0, 0, plot_size_x, plot_size_x, 0], [0, plot_size_y, plot_size_y, 0, 0], [0], label='walls')

    # floor
    floor_x, floor_y = np.meshgrid(np.arange(0, plot_size_x + 1), np.arange(0, plot_size_y + 1))
    floor_z = np.zeros((plot_size_x + 1, plot_size_y + 1), dtype=np.int)
    ax.plot_surface(floor_x, floor_y, floor_z, alpha=0.2)

    # cameras
    plotted_cameras = []
    for view in person_time_frame.views:
        if view.camera not in plotted_cameras:
            plotted_cameras.append(view.camera)
            # camera point
            ax.scatter(
                [view.camera.position[0]],
                [view.camera.position[1]],
                [view.camera.position[2]],
                label=view.camera.name
            )
            # camera ray
            ax.plot(
                [view.camera.position[0], view.camera.position[0] + view.camera.orientation[0] * ray_extender],
                [view.camera.position[1], view.camera.position[1] + view.camera.orientation[1] * ray_extender],
                [view.camera.position[2], view.camera.position[2] + view.camera.orientation[2] * ray_extender],
                label=view.camera.name + ' ray'
            )

    # distance planes
    for plane in person_time_frame.distance_planes:
        plane_z, plane_x = np.meshgrid(np.arange(0, plot_size_z + 1), np.arange(0, distance_plane_size_xy + 1))
        plane_y = np.array([[plane(z, x) for z in range(plot_size_z + 1)] for x in range(distance_plane_size_xy + 1)])
        ax.plot_surface(plane_x, plane_y, plane_z, alpha=0.2)

    # intersection line
    if person_time_frame.intersection_line is not None:
        point = person_time_frame.intersection_line[0]
        vector = person_time_frame.intersection_line[1]
        point_forwards = point + vector * intersection_line_extender
        point_backwards = point - vector * intersection_line_extender
        ax.plot(
            [point[0], point_forwards[0], point_backwards[0]],
            [point[1], point_forwards[1], point_backwards[1]],
            [point[2], point_forwards[2], point_backwards[2]],
            label='intersection line'
        )

    # real subject
    if person_time_frame.real_subject_coordinates_3d is not None:
        ax.scatter(
            [person_time_frame.real_subject_coordinates_3d[0]],
            [person_time_frame.real_subject_coordinates_3d[1]],
            [person_time_frame.real_subject_coordinates_3d[2]],
            label='real subject'
        )

    # located subject
    if person_time_frame.coordinates_3d is not None:
        ax.scatter(
            [person_time_frame.coordinates_3d[0]],
            [person_time_frame.coordinates_3d[1]],
            [person_time_frame.coordinates_3d[2]],
            label='located subject'
        )

    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')
    ax.legend()
    ax.set_zlim(0, plot_size_z)
    plt.show()


class Person:
    """
    Unique person with time frames of its detections.
    """

    def __init__(self, time_frame: PersonTimeFrame, name: str = None):
        if name is None:
            name = "unknown person id={0}".format(id(self))
        self.name = name
        self.time_frames = [time_frame]
