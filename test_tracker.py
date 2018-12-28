"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""

from unittest import TestCase

from camera import Camera
from config import FOCAL_LENGTH_CAMERA_M, FOCAL_LENGTH_CAMERA_F, AVERAGE_PERSON_WAIST_TO_NECK_LENGTH
from detector import OpenPoseDetector, PeopleDetector
from image_provider import DummyImageProvider, ImageProvider
from matcher import HistogramMatcher, PersonMatcher
from tracker import HistogramTracker, PersonTracker
from triangulation import CameraDistanceTriangulation, Triangulation


class TestHistogramTracker(TestCase):

    def setUp(self):
        z_level = 147
        self.camera_front = Camera(
            name='front camera (m)',
            focal_length=FOCAL_LENGTH_CAMERA_M,
            position=(0, 0, z_level),
            orientation=(0, 1, 0)
        )
        self.camera_side = Camera(
            name='side camera (f)',
            focal_length=FOCAL_LENGTH_CAMERA_F,
            position=(-200, 0, z_level),
            orientation=(1, 1, 0)
        )
        prototxt_path = "openpose/pose/coco/pose_deploy_linevec.prototxt"
        caffemodel_path = "openpose/pose/coco/pose_iter_440000.caffemodel"
        self.detector = OpenPoseDetector(prototxt_path, caffemodel_path)  # type: PeopleDetector
        self.matcher = HistogramMatcher()  # type: PersonMatcher
        # type: Triangulation
        self.triangulation = CameraDistanceTriangulation(AVERAGE_PERSON_WAIST_TO_NECK_LENGTH, z_level)
        print('setUp finished')

    def prepare_frames(self, front_image, side_image):
        print('prepare_frames started')
        front_views = self.detector.detect(front_image, self.camera_front)
        side_views = self.detector.detect(side_image, self.camera_side)
        time_frames = self.matcher.match(front_views, side_views)
        time_frames_located = []
        for frame in time_frames:
            located_frame = self.triangulation.locate(frame)
            time_frames_located.append(located_frame)
        print('prepare_frames finished')
        return time_frames_located

    def test_track(self):
        # first frame
        image_provider = DummyImageProvider(front_image_path='testing_data/s3_m_front_single_x0y300.png',
                                            side_image_path='testing_data/s3_f_side_single_x0y300.png',
                                            )  # type: ImageProvider
        front, side = next(image_provider)
        time_frames_located = self.prepare_frames(front, side)
        self.assertEqual(1, len(time_frames_located), 'prepare_frames failed to prepare correct number of frames')
        time_frame = time_frames_located[0]

        tracker = HistogramTracker()
        self.assertEqual(0, len(tracker.people), 'New tracker should have empty list of tracked people.')
        person = tracker.track(time_frame)
        self.assertEqual(1, len(tracker.people))  # first person was tracked

        person2 = tracker.track(time_frame)  # try to track *exactly* the same frame again
        self.assertEqual(person, person2)
        self.assertEqual(1, len(tracker.people))

        # second frame
        image_provider = DummyImageProvider(front_image_path='testing_data/s3_m_front_single_x50y600.png',
                                            side_image_path='testing_data/s3_f_side_single_x50y600.png',
                                            )  # type: ImageProvider
        front, side = next(image_provider)
        time_frames_located = self.prepare_frames(front, side)
        self.assertEqual(1, len(time_frames_located), 'prepare_frames failed to prepare correct number of frames')
        time_frame = time_frames_located[0]

        person3 = tracker.track(time_frame)  # track second frame
        self.assertEqual(person, person3)
        self.assertEqual(1, len(tracker.people))
