#!/usr/bin/env python3
"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology

Basic triangulation from multiple cameras.

Notes:
    http://answers.opencv.org/question/117141/triangulate-3d-points-from-a-stereo-camera-and-chessboard/
"""
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple


class Camera:
    def __init__(self, name: str, focal_length: int, position: Tuple[int, int, int], orientation: Tuple[int, int, int]):
        """
        :param name: camera name used in user interface
        :param focal_length: focal length of camera, see `Camera.calibrate_focal_length`
        :param position: 3D coordinates of the camera position
        :param orientation: 3D vector of the camera's view orientation
        """
        self.name = name
        self.focal_length = focal_length
        # TODO Position in a specific scene?
        self.position = position
        self.orientation = orientation / np.linalg.norm(orientation)  # convert to unit vector
        # TODO store calibrated parameters of this camera, which are used for triangulation

    def calibrate_focal_length(self, real_distance: int, real_size: int, pixel_size: int):
        """
        Calibrate focal length of the camera based on measurement of known object and its image representation.
        :param real_distance: real distance of known object from camera in millimeters
        :param real_size: real size size of known object in millimeters,
        :param pixel_size: size of known object measured in pixels in image obtained from the camera
        """
        self.focal_length = (pixel_size * real_distance) / real_size


class PersonView:
    """
    Single view of a person detected in an image from given camera.
    View includes detected position of person's pose.
    """

    def __init__(self, camera_image, camera: Camera, pose_top_coordinate, pose_bottom_coordinate):
        self.camera_image = camera_image
        self.camera = camera
        self.pose_top_coordinate = pose_top_coordinate
        self.pose_bottom_coordinate = pose_bottom_coordinate


class PersonTimeFrame:
    """
    Time frame when given person was detected on at least one view of the scene.
    """

    def __init__(self, time, views: List[PersonView]):
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
    floor_x, floor_y = np.meshgrid(np.arange(0, plot_size_x+1), np.arange(0, plot_size_y+1))
    floor_z = np.zeros((plot_size_x+1, plot_size_y+1), dtype=np.int)
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
        plane_z, plane_x = np.meshgrid(np.arange(0, plot_size_z+1), np.arange(0, distance_plane_size_xy+1))
        plane_y = np.array([[plane(z, x) for z in range(plot_size_z+1)] for x in range(distance_plane_size_xy+1)])
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

    def __init__(self, time_frames: List[PersonTimeFrame]):
        self.time_frames = time_frames


class Triangulation(ABC):
    @abstractmethod
    def locate(self, person_time_frame: PersonTimeFrame) -> PersonTimeFrame:
        """
        Locate person in space based on provided time frame with views from cameras.
        Sets person_time_frame.coordinates_3d on success.
        """
        pass


class CameraDistanceTriangulation(Triangulation):
    """
    Triangulation of subject's position based on distance from two cameras.
    """
    def __init__(self, real_size: int, z_location: int):
        """
        :param real_size: real size of subject which cameras are calibrated to
        :param z_location: fixed z coordinate on which the subject should be located
        """
        # TODO: It would be better not to use average human size. Maybe implement different triangulation method.
        self.real_size = real_size
        self.z_location = z_location

    def locate(self, person_time_frame: PersonTimeFrame) -> PersonTimeFrame:
        if len(person_time_frame.views) != 2:
            raise NotImplementedError('triangulation currently supports only two cameras per time frame')
        planes_analytical = []
        # Get distance planes based on distance from cameras and cameras' orientation.
        # Subject is somewhere in distance planes.
        for view in person_time_frame.views:
            # distance plane: ax + by + cz + d = 0; point = (x, y, z); normal = (a, b, c)
            point = view.camera.position + view.camera.orientation * self.distance_from_camera(view)
            normal = view.camera.orientation
            d = - point.dot(normal)
            planes_analytical.append((normal[0], normal[1], normal[2], d))
            person_time_frame.distance_planes.append(  # TODO This is used only for visualization
                # NOTE: Python lambda's binding to local values
                # https://stackoverflow.com/questions/10452770/python-lambdas-binding-to-local-values
                lambda z, x, a=normal[0], b=normal[1], c=normal[2]: (- c * z - a * x - d) / b
            )
        # Get intersection of distance planes. Subject is somewhere on the intersection line.
        point, vector = self.intersect_planes(planes_analytical[0], planes_analytical[1])
        person_time_frame.intersection_line = (point, vector)  # TODO This is used only for visualization

        # Locate subject on desired z level based on `self.z_location`.
        # line: X = A + tu; A = point = (a0, a1, a2); u = vector = (u0, u1, u2)
        # x = a0 + t * u0
        # y = a1 + t * u1
        # z = a2 + t * u2
        t = (self.z_location - point[2]) / vector[2]
        subject_x = point[0] + t * vector[0]
        subject_y = point[1] + t * vector[1]
        person_time_frame.coordinates_3d = (subject_x, subject_y, self.z_location)
        return person_time_frame

    @staticmethod
    def intersect_planes(plane_a, plane_b):
        """
        Takes tuple (a,b,c,d) of each plane `ax + by + cz + d = 0`.
        https://stackoverflow.com/questions/48126838/plane-plane-intersection-in-python
        """
        a_normal = np.array(plane_a[:3])
        b_normal = np.array(plane_b[:3])
        intersection_vector = np.cross(a_normal, b_normal)
        a = np.array([a_normal, b_normal, intersection_vector])
        d = np.array([-plane_a[3], -plane_b[3], 0.]).reshape(3, 1)
        intersection = np.linalg.solve(a, d).T
        return intersection[0], intersection_vector

    def distance_from_camera(self, person_view: PersonView) -> int:
        current_size = math.hypot(person_view.pose_top_coordinate[0] - person_view.pose_bottom_coordinate[0],
                                  person_view.pose_top_coordinate[1] - person_view.pose_bottom_coordinate[1],
                                  )
        return int((self.real_size * person_view.camera.focal_length) / current_size)
