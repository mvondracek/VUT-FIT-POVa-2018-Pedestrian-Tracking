"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import datetime
from typing import List

import cv2
import numpy as np

from camera import Camera
from tools import euclidean_distance


class PersonView:
    """
    Single view of a person detected in an image from given camera.
    View includes detected position of person's pose.
    """

    def __init__(self, original_image, person_image, camera: Camera, pose_top_coordinate, pose_bottom_coordinate):
        """
        :param original_image: original full-size image from camera
        :param person_image: image containing this person
        """
        self.original_image = original_image
        self.person_image = person_image
        self.camera = camera
        self.pose_top_coordinate = pose_top_coordinate
        self.pose_bottom_coordinate = pose_bottom_coordinate

    def get_torso_subimage(self):
        """
        Extract a subimage of just a torso for given person view.
        Should be better for histograms since contains less surroundings than the whole person box.
        :return: subimage containing only the torso
        """
        image_width = self.original_image.shape[1]
        body_height = int(euclidean_distance(self.pose_top_coordinate, self.pose_bottom_coordinate))
        # an average body's width from side is about one third of the height (half of the height from front)
        half_body_width = int(body_height / 6)

        pose_top_left = (min(self.pose_top_coordinate[0], self.pose_bottom_coordinate[0]),
                         min(self.pose_top_coordinate[1], self.pose_bottom_coordinate[1]))
        pose_bottom_right = (max(self.pose_top_coordinate[0], self.pose_bottom_coordinate[0]),
                             max(self.pose_top_coordinate[1], self.pose_bottom_coordinate[1]))

        roi_top_left = (max(0, pose_top_left[0] - half_body_width), pose_top_left[1])
        roi_bottom_right = (min(image_width, pose_bottom_right[0] + half_body_width), pose_bottom_right[1])

        return self.original_image[roi_top_left[1]:roi_bottom_right[1] + 1, roi_top_left[0]:roi_bottom_right[0] + 1]

    def show(self):
        # FIXME display the whole image and the person inside it? TODO change person subimage to mask
        window_name = 'PersonView: {}'.format(self.__hash__())
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, self.person_image)


class PersonTimeFrame:
    """
    Time frame when given person was detected on at least one view of the scene.
    """

    def __init__(self, views: List[PersonView], time=datetime.datetime.now()):
        self.time = time
        self.views = views
        self.coordinates_3d = None

        self.distance_planes = []  # TODO This is used only for visualization of planes (debug)
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
    plt.show()


class Person:
    """
    Unique person with time frames of its detections.
    """

    def __init__(self, time_frame: PersonTimeFrame, name: str = None):
        if name is None:
            name = "person id={0}".format(id(self))
        self.name = name
        self.time_frames = [time_frame]
