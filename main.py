#!/usr/bin/env python3
"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import logging
import sys
import threading
import warnings
from enum import unique, Enum

import coloredlogs as coloredlogs
import cv2
import matplotlib.pyplot as plt

from camera import Camera
from config import FOCAL_LENGTH_CAMERA_M, FOCAL_LENGTH_CAMERA_F, AVERAGE_PERSON_WAIST_TO_NECK_LENGTH
from detector import OpenPoseDetector, PeopleDetector
from image_provider import ImageProvider, ImageProviderFromVideo
from matcher import PersonMatcher, HistogramMatcher
from tracker import NullTracker, PersonTracker, HistogramTracker
from triangulation import CameraDistanceTriangulation, Triangulation
from visualizer import Plotter3D, Visualizer

logger = logging.getLogger(__name__)


@unique
class ExitCode(Enum):
    """
    Exit codes. Some are inspired by sysexits.h.
    """
    EX_OK = 0
    """Program terminated successfully."""

    ARGUMENTS = 2
    """Incorrect or missing arguments provided."""

    EX_UNAVAILABLE = 69
    """Required program or file does not exist."""

    EX_NOPERM = 77
    """Permission denied."""

    KEYBOARD_INTERRUPT = 130
    """Program received SIGINT."""


def main() -> ExitCode:
    logging.captureWarnings(True)
    warnings.simplefilter('always', ResourceWarning)
    coloredlogs.install(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    logger.debug('main started')

    # region Initialization
    logger.debug('initializing pipeline')
    z_level = 147
    camera_front = Camera(
        name='front camera (m)',
        focal_length=FOCAL_LENGTH_CAMERA_M,
        position=(0, 0, z_level),
        orientation=(0, 1, 0)
    )
    camera_side = Camera(
        name='side camera (f)',
        focal_length=FOCAL_LENGTH_CAMERA_F,
        position=(-200, 0, z_level),
        orientation=(1, 1, 0)
    )
    prototxt_path = "openpose/pose/coco/pose_deploy_linevec.prototxt"
    caffemodel_path = "openpose/pose/coco/pose_iter_440000.caffemodel"
    image_provider = ImageProviderFromVideo(
         ['testing_data/s3_m_front_single.mov', 'testing_data/s3_f_side_single.mov'],
         start=39*30,  # start after first few seconds # used for s3_m_front_single.mov and s3_f_side_single.mov
         #['testing_data/s3_m_front_multi.mov', 'testing_data/s3_f_side_multi.mov'],
         #start=43*30,  # start after first few seconds # used for s3_m_front_multi.mov and s3_f_side_multi.mov
         skipping=30)  # type: ImageProvider # (30 fps)
    detector = OpenPoseDetector(prototxt_path, caffemodel_path)  # type: PeopleDetector
    matcher = HistogramMatcher()  # type: PersonMatcher
    triangulation = CameraDistanceTriangulation(AVERAGE_PERSON_WAIST_TO_NECK_LENGTH, z_level)  # type: Triangulation
    tracker = HistogramTracker()  # type: PersonTracker
    visualizer = Plotter3D(tracker.people, [camera_front, camera_side])  # type: Visualizer
    # endregion

    def processing_pipeline():
        for i, image_set in enumerate(image_provider):
            logger.info('step {}'.format(i))
            front_image, side_image = image_set

            logger.info('detecting people')
            front_views = detector.detect(front_image, camera_front)
            side_views = detector.detect(side_image, camera_side)

            logger.info('matching people')
            time_frames = matcher.match(front_views, side_views)

            logger.info('locating people')
            time_frames_located = []
            for time_frame in time_frames:
                located_frame = triangulation.locate(time_frame)
                time_frames_located.append(located_frame)

            logger.info('tracking people')
            for time_frame in time_frames_located:
                person = tracker.track(time_frame)
                logger.info("Person={}, 3D={}".format(person.name, person.time_frames[-1].coordinates_3d))

    processing_thread = threading.Thread(target=processing_pipeline)
    processing_thread.start()

    while True:
        visualizer.render()
        plt.pause(2)
        plt.show()
        if cv2.waitKey(2) & 0xFF == ord('q'):
            logger.debug('break')
            break

    # TODO: Better thread handling. Better application shutdown.
    processing_thread.join(timeout=10)

    return ExitCode.EX_OK


if __name__ == '__main__':
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print('Stopping.')
        logger.warning('received KeyboardInterrupt, stopping')
        sys.exit(ExitCode.KEYBOARD_INTERRUPT.value)
    else:
        sys.exit(exit_code)
