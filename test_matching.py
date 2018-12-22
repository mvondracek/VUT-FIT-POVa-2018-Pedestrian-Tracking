"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import logging
from unittest import TestCase

import cv2

from camera import Camera
from config import FOCAL_LENGTH_CAMERA_M, FOCAL_LENGTH_CAMERA_F
from detector import OpenPoseDetector
from matcher import HistogramMatcher

logger = logging.getLogger(__name__)


class MatchingTestCase(TestCase):
    front_image_single = cv2.imread('testing_data/s3_m_front_single_x0y300.png')
    side_image_single = cv2.imread('testing_data/s3_f_side_single_x0y300.png')
    front_image_multi = cv2.imread('testing_data/s3_m_front_multi_y600.png')
    side_image_multi = cv2.imread('testing_data/s3_f_side_multi_y600.png')

    front_camera = Camera(name='m (front camera)', focal_length=FOCAL_LENGTH_CAMERA_M, position=(0, 0, 147), orientation=(0, 1, 0))
    side_camera = Camera(name='f (side camera)', focal_length=FOCAL_LENGTH_CAMERA_F, position=(200, 0, 147), orientation=(-1, 1, 0))

    detector = OpenPoseDetector("openpose/pose/coco/pose_deploy_linevec.prototxt", "openpose/pose/coco/pose_iter_440000.caffemodel")


class TestHistogramMatcher(MatchingTestCase):
    def setUp(self) -> None:
        self.matcher = HistogramMatcher()

    def test_single_person_matching(self):
        logger.debug('Detecting people...')
        front_views = self.detector.detect(self.front_image_single, self.front_camera)
        side_views = self.detector.detect(self.side_image_single, self.side_camera)

        logger.debug('Matching single person')
        self.matcher.set_original_images(self.front_image_single, self.side_image_single)
        time_frames = self.matcher.match(front_views, side_views)

        self.assertEqual(len(time_frames), 1, "Matched {} people. Expected 1 person!".format(len(time_frames)))
        self.assertSequenceEqual(time_frames[0].views, [front_views[0], side_views[0]], "Matched unexpected person: {}".format(time_frames[0].views))

    def test_multiple_person_matching(self):
        logger.debug('Detecting people...')
        front_views = self.detector.detect(self.front_image_multi, self.front_camera)
        side_views = self.detector.detect(self.side_image_multi, self.side_camera)

        logger.debug('Matching single person')
        self.matcher.set_original_images(self.front_image_multi, self.side_image_multi)
        time_frames = self.matcher.match(front_views, side_views)

        expected_matches = 3
        expected_pairs = ([front_views[0], side_views[1]], [front_views[1], side_views[2]], [front_views[2], side_views[0]])
        self.assertEqual(len(time_frames), expected_matches, "Matched {0} people. Expected {1} people!".format(len(time_frames), expected_matches))
        self.assertSequenceEqual(time_frames[0].views, expected_pairs[0], "Unexpected match for person 1!")
        self.assertSequenceEqual(time_frames[1].views, expected_pairs[1], "Unexpected match for person 2!")
        self.assertSequenceEqual(time_frames[2].views, expected_pairs[2], "Unexpected match for person 3!")
