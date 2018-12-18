#!/usr/bin/env python3
"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import math

import cv2
from unittest import TestCase

import openpose


class TestPovaPose(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.person_detector = openpose.PovaPose.PovaPose(
            prototxt_path="pose/coco/pose_deploy_linevec.prototxt",
            caffemodel_path="pose/coco/pose_iter_440000.caffemodel"
        )

    def test_run_multi_person_detection(self):
        self.person_detector.set_image_for_detection(cv2.imread('../testing_data/s2_m_x0y300.png'))
        people = self.person_detector.run_multi_person_detection()
        self.assertEqual(len(people), 1, "Detected incorrect number of people.")
        # neck
        self.assertAlmostEqual(people[0][1][0], 854, delta=10)
        self.assertAlmostEqual(people[0][1][1], 551, delta=10)
        # waist
        self.assertAlmostEqual(people[0][2][0], 843, delta=10)
        self.assertAlmostEqual(people[0][2][1], 880, delta=10)
        # self.person_detector.show()

        self.person_detector.set_image_for_detection(cv2.imread('../testing_data/s2_m_x0y600.png'))
        people = self.person_detector.run_multi_person_detection()
        self.assertEqual(len(people), 1, "Detected incorrect number of people.")
        # neck
        self.assertAlmostEqual(people[0][1][0], 947, delta=10)
        self.assertAlmostEqual(people[0][1][1], 551, delta=10)
        # waist
        self.assertAlmostEqual(people[0][2][0], 936, delta=10)
        self.assertAlmostEqual(people[0][2][1], 716, delta=10)
        # self.person_detector.show()

        self.person_detector.set_image_for_detection(cv2.imread('../testing_data/s2_f_x0y300.png'))
        people = self.person_detector.run_multi_person_detection()
        self.assertEqual(len(people), 1, "Detected incorrect number of people.")
        # neck
        self.assertAlmostEqual(people[0][1][0], 900, delta=10)
        self.assertAlmostEqual(people[0][1][1], 526, delta=10)
        # waist
        self.assertAlmostEqual(people[0][2][0], 877, delta=10)
        self.assertAlmostEqual(people[0][2][1], 867, delta=10)
        # self.person_detector.show()

        self.person_detector.set_image_for_detection(cv2.imread('../testing_data/s2_f_x0y600.png'))
        people = self.person_detector.run_multi_person_detection()
        self.assertEqual(len(people), 1, "Detected incorrect number of people.")
        # neck
        self.assertAlmostEqual(people[0][1][0], 972, delta=10)
        self.assertAlmostEqual(people[0][1][1], 528, delta=10)
        # waist
        self.assertAlmostEqual(people[0][2][0], 983, delta=10)
        self.assertAlmostEqual(people[0][2][1], 692, delta=10)
        # self.person_detector.show()

    def test_waist_to_neck(self):
        self.person_detector.set_image_for_detection(cv2.imread('../testing_data/s2_f_x0y300.png'))
        people = self.person_detector.run_multi_person_detection()
        # self.person_detector.show()
        waist_to_neck = length(people[0][1], people[0][2])  # real=53
        self.assertAlmostEqual(waist_to_neck, 341, delta=10)

        self.person_detector.set_image_for_detection(cv2.imread('../testing_data/s2_f_x0y600.png'))
        people = self.person_detector.run_multi_person_detection()
        # self.person_detector.show()
        waist_to_neck = length(people[0][1], people[0][2])
        self.assertAlmostEqual(waist_to_neck, 164, delta=10)

        self.person_detector.set_image_for_detection(cv2.imread('../testing_data/s2_m_x0y300.png'))
        people = self.person_detector.run_multi_person_detection()
        # self.person_detector.show()
        waist_to_neck = length(people[0][1], people[0][2])
        self.assertAlmostEqual(waist_to_neck, 329, delta=10)

        self.person_detector.set_image_for_detection(cv2.imread('../testing_data/s2_m_x0y600.png'))
        people = self.person_detector.run_multi_person_detection()
        # self.person_detector.show()
        waist_to_neck = length(people[0][1], people[0][2])
        self.assertAlmostEqual(waist_to_neck, 165, delta=10)


def length(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1], )
