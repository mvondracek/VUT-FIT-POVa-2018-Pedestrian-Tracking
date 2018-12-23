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

import openpose
from camera import Camera
from config import FOCAL_LENGTH_CAMERA_M, FOCAL_LENGTH_CAMERA_F, AVERAGE_PERSON_WAIST_TO_NECK_LENGTH
from triangulation import CameraDistanceTriangulation, PersonView, PersonTimeFrame


class TestCameraDistanceTriangulation(TestCase):
    def test_intersect_planes(self):
        """
        https://stackoverflow.com/questions/48126838/plane-plane-intersection-in-python
        """
        a = (1, -1, 0, 2)
        b = (-1, -1, 1, 3)
        point, vector = CameraDistanceTriangulation.intersect_planes(a, b)
        point_forward = point + vector
        a_normal = np.array(a[:3])
        b_normal = np.array(b[:3])
        self.assertEqual(np.dot(point, a_normal), np.dot(point_forward, a_normal))
        self.assertEqual(np.dot(point, b_normal), np.dot(point_forward, b_normal))


class TestCameraDistanceTriangulationScene1Duck(TestCase):
    """
    Test triangulation based on distance from camera in scene1 (duck).
    """

    def setUp(self) -> None:
        super().setUp()
        focal_length = 687  # see Camera.calibrate_focal_length
        self.real_size = 50
        camera_front = Camera('front camera', focal_length, (50, 450, 30), (0, -1, 0))
        camera_side = Camera('side camera', focal_length, (400, 400, 30), (-1, -1, 0))

        # reference = PersonView(cv2.imread('testing_data/s1_front_d150_h50.jpg'), camera_front, (379, 70), (379, 299))
        self.front = PersonView(cv2.imread('testing_data/s1_front_d400.jpg'), camera_front, (383, 95), (383, 11))
        self.side = PersonView(cv2.imread('testing_data/s1_side_d500.jpg'), camera_side, (77, 86), (77, 18))

        self.person_time_frame = PersonTimeFrame([self.front, self.side])
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


class TestCameraDistanceTriangulationSceneLibrary(TestCase):
    """
    Test triangulation based on distance from camera in scene Library.
    """

    def setUp(self) -> None:
        super().setUp()
        self.real_size = AVERAGE_PERSON_WAIST_TO_NECK_LENGTH  # cm
        z_level = 146
        self.camera_m = Camera(
            name='m (front camera)',
            focal_length=FOCAL_LENGTH_CAMERA_M,
            position=(0, 0, z_level),
            orientation=(0, 1, 0)
        )
        self.camera_f = Camera(
            name='f (front camera)',
            focal_length=FOCAL_LENGTH_CAMERA_F,
            position=(0, 0, z_level),
            orientation=(0, 1, 0)
        )
        self.person_detector = openpose.PovaPose.PovaPose(
            prototxt_path="openpose/pose/coco/pose_deploy_linevec.prototxt",
            caffemodel_path="openpose/pose/coco/pose_iter_440000.caffemodel"
        )
        self.triangulation = CameraDistanceTriangulation(self.real_size, z_location=z_level)

    def test_distance_from_camera(self):
        def assert_distance(camera, image_path, distance, delta):
            image = cv2.imread(image_path)
            self.person_detector.set_image_for_detection(image)
            people = self.person_detector.run_multi_person_detection()
            self.assertEqual(len(people), 1, "Detected incorrect number of people.")
            person = people[0]
            view = PersonView(image, camera, (person[1][1], person[1][0]), (person[2][1], person[2][0]))
            self.assertAlmostEqual(self.triangulation.distance_from_camera(view), distance, delta=delta)

        assert_distance(self.camera_f, 'testing_data/s2_f_x0y300.png', 300, 15)
        assert_distance(self.camera_f, 'testing_data/s2_f_x0y600.png', 600, 15)
        assert_distance(self.camera_f, 'testing_data/s2_m_x0y300.png', 300, 15)
        assert_distance(self.camera_f, 'testing_data/s2_m_x0y600.png', 600, 15)


class TestCameraDistanceTriangulationSceneCorridor(TestCase):
    """
    Test triangulation based on distance from camera in scene Corridor.
    """

    def setUp(self) -> None:
        super().setUp()
        self.real_size = AVERAGE_PERSON_WAIST_TO_NECK_LENGTH  # cm
        self.camera_m = Camera(
            name='m (front camera)',
            focal_length=FOCAL_LENGTH_CAMERA_M,
            position=(0, 0, 147),
            orientation=(0, 1, 0)
        )
        self.camera_f = Camera(
            name='f (side camera)',
            focal_length=FOCAL_LENGTH_CAMERA_F,
            position=(200, 0, 147),
            orientation=(-1, 1, 0)
        )
        self.person_detector = openpose.PovaPose.PovaPose(
            prototxt_path="openpose/pose/coco/pose_deploy_linevec.prototxt",
            caffemodel_path="openpose/pose/coco/pose_iter_440000.caffemodel"
        )
        self.triangulation = CameraDistanceTriangulation(self.real_size, z_location=147)

    def test_distance_from_camera(self):
        def assert_distance(camera, image_path, distance, delta):
            image = cv2.imread(image_path)
            self.person_detector.set_image_for_detection(image)
            people = self.person_detector.run_multi_person_detection()
            self.assertEqual(len(people), 1, "Detected incorrect number of people.")
            person = people[0]
            view = PersonView(image, camera, (person[1][1], person[1][0]), (person[2][1], person[2][0]))
            self.assertAlmostEqual(self.triangulation.distance_from_camera(view), distance, delta=delta)

        assert_distance(self.camera_m, 'testing_data/s3_m_front_single_x0y300.png', 300, 20)
        assert_distance(self.camera_m, 'testing_data/s3_m_front_single_x-50y600.png', 600, 20)

    def test_locate(self):
        def create_person_view(camera, image_path):
            image_front = cv2.imread(image_path)
            self.person_detector.set_image_for_detection(image_front)
            people = self.person_detector.run_multi_person_detection()
            self.assertEqual(len(people), 1, "Detected incorrect number of people.")
            person = people[0]
            return PersonView(image_front, camera, (person[1][1], person[1][0]), (person[2][1], person[2][0]))

        # distance 300
        front_view = create_person_view(self.camera_m, 'testing_data/s3_m_front_single_x0y300.png')
        side_view = create_person_view(self.camera_f, 'testing_data/s3_f_side_single_x0y300.png')

        person_time_frame = PersonTimeFrame([front_view, side_view])
        person_time_frame.real_subject_coordinates_3d = (0, 300, 147)

        located = self.triangulation.locate(person_time_frame)
        # plot_person_time_frame(located) # for debugging
        self.assertAlmostEqual(located.coordinates_3d[0], located.real_subject_coordinates_3d[0], delta=30)
        self.assertAlmostEqual(located.coordinates_3d[1], located.real_subject_coordinates_3d[1], delta=30)
        self.assertAlmostEqual(located.coordinates_3d[2], located.real_subject_coordinates_3d[2], delta=30)

        # distance 600
        front_view = create_person_view(self.camera_m, 'testing_data/s3_m_front_single_x-50y600.png')
        side_view = create_person_view(self.camera_f, 'testing_data/s3_f_side_single_x-50y600.png')

        person_time_frame = PersonTimeFrame([front_view, side_view])
        person_time_frame.real_subject_coordinates_3d = (0, 600, 147)

        located = self.triangulation.locate(person_time_frame)
        # plot_person_time_frame(located) # for debugging
        self.assertAlmostEqual(located.coordinates_3d[0], located.real_subject_coordinates_3d[0], delta=30)
        self.assertAlmostEqual(located.coordinates_3d[1], located.real_subject_coordinates_3d[1], delta=30)
        self.assertAlmostEqual(located.coordinates_3d[2], located.real_subject_coordinates_3d[2], delta=30)
