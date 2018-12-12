#!/usr/bin/env python3
"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import datetime
from unittest import TestCase

import cv2
import numpy as np

from triangulation import CameraDistanceTriangulation, Camera, PersonView, PersonTimeFrame


class TestCameraDistanceTriangulation(TestCase):

    def setUp(self) -> None:
        super().setUp()
        focal_length = 687  # see Camera.calibrate_focal_length
        self.real_size = 50
        camera_front = Camera('front camera', focal_length, (50, 450, 30), (0, -1, 0))
        camera_side = Camera('side camera', focal_length, (400, 400, 30), (-1, -1, 0))

        # reference = PersonView(cv2.imread('testing_data/s1_front_d150_h50.jpg'), camera_front, (379, 70), (379, 299))
        self.front = PersonView(cv2.imread('testing_data/s1_front_d400.jpg'), camera_front, (383, 95), (383, 11))
        self.side = PersonView(cv2.imread('testing_data/s1_side_d500.jpg'), camera_side, (77, 86), (77, 18))

        self.person_time_frame = PersonTimeFrame(datetime.datetime.now(), [self.front, self.side])
        self.person_time_frame.real_subject_coordinates_3d = (50, 50, 30)
        # person = Person([person_time_frame])

    def test_locate(self):
        triangulation = CameraDistanceTriangulation(self.real_size, 30)
        located = triangulation.locate(self.person_time_frame)
        # plot_person_time_frame(located) # for debugging
        self.assertAlmostEqual(located.coordinates_3d[0], located.real_subject_coordinates_3d[0], delta=10)
        self.assertAlmostEqual(located.coordinates_3d[1], located.real_subject_coordinates_3d[1], delta=10)
        self.assertAlmostEqual(located.coordinates_3d[2], located.real_subject_coordinates_3d[2], delta=10)

    def test_distance_from_camera(self):
        # plot_person_time_frame(self.person_time_frame)  # for debugging
        triangulation = CameraDistanceTriangulation(self.real_size, 30)
        self.assertAlmostEqual(triangulation.distance_from_camera(self.front), 400, delta=10)
        self.assertAlmostEqual(triangulation.distance_from_camera(self.side), 500, delta=10)

    def test_intersect_planes(self):
        """
        https://stackoverflow.com/questions/48126838/plane-plane-intersection-in-python
        """
        a = (1, -1, 0, 2)
        b = (-1, -1, 1, 3)
        triangulation = CameraDistanceTriangulation(self.real_size, 30)
        point, vector = triangulation.intersect_planes(a, b)
        point_forward = point + vector
        a_normal = np.array(a[:3])
        b_normal = np.array(b[:3])
        self.assertEqual(np.dot(point, a_normal), np.dot(point_forward, a_normal))
        self.assertEqual(np.dot(point, b_normal), np.dot(point_forward, b_normal))
