"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology

Basic triangulation from multiple cameras.

Notes:
    http://answers.opencv.org/question/117141/triangulate-3d-points-from-a-stereo-camera-and-chessboard/
    https://github.com/mbarde/twocams/blob/master/README.md
    http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
"""
import math
from abc import ABC, abstractmethod

import numpy as np

from person import PersonTimeFrame, PersonView


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
