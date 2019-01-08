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
import multiprocessing
import warnings
from enum import unique, Enum
from typing import List

import coloredlogs as coloredlogs
import cv2
import matplotlib.pyplot as plt
import numpy as np

from camera import Camera
from config import FOCAL_LENGTH_CAMERA_M, FOCAL_LENGTH_CAMERA_F, AVERAGE_PERSON_WAIST_TO_NECK_LENGTH
from detector import OpenPoseDetector, PeopleDetector
from image_provider import ImageProvider, ImageProviderFromVideo, DummyImageProvider
from image_tweaks import ImgTweaksBasedOnPresumablySameArea
from matcher import PersonMatcher, HistogramMatcher
from tracker import PersonTracker, HistogramTracker
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


def processing_pipeline(cameras: List[Camera], z_level: int, output_queue: multiprocessing.Queue):
    """
    :param cameras: cameras in order as their corresponding images will be used
    :param z_level: cameras height
    :param output_queue: <multiprocessing.Queue> results (tracked people list) are put to this Q after each step
    :return:
    """
    # TODO extract this to a class, so it can be like Pipeline -> ProjectPipeline(Pipeline) with run() method
    # region Initialization
    logger.debug('initializing pipeline')
    prototxt_path = "openpose/pose/coco/pose_deploy_linevec.prototxt"
    caffemodel_path = "openpose/pose/coco/pose_iter_440000.caffemodel"
    # create masks to config image tweaks (without need of user interaction)
    f_rect = ([1443, 428], [1626, 481])
    s_rect = ([373, 346], [534, 389])
    mask_shape = (1080, 1920)
    f_mask = np.zeros(mask_shape, dtype=np.uint8)  # mask needs to has one dimension less
    f_mask[f_rect[0][1]:f_rect[1][1] + 1, f_rect[0][0]:f_rect[1][0] + 1] = 1
    s_mask = np.zeros(mask_shape, dtype=np.uint8)  # mask needs to has one dimension less
    s_mask[s_rect[0][1]:s_rect[1][1] + 1, s_rect[0][0]:s_rect[1][0] + 1] = 1
    image_tweaker = ImgTweaksBasedOnPresumablySameArea(interactive=False, masks=[f_mask, s_mask])
    image_provider = ImageProviderFromVideo(
        # ['testing_data/s3_m_front_single.mov', 'testing_data/s3_f_side_single.mov'],
        # start=39*30,  # start after first few seconds # used for s3_m_front_single.mov and s3_f_side_single.mov
        ['testing_data/s3_m_front_multi.mov', 'testing_data/s3_f_side_multi.mov'],
        start=43*30,  # start after first few seconds # used for s3_m_front_multi.mov and s3_f_side_multi.mov
        skipping=10,
        image_tweaker=image_tweaker)  # type: ImageProvider # (30 fps)

    # image_provider = DummyImageProvider(front_image_path='testing_data/s3_m_front_single_x0y300.png',
    #                                     side_image_path='testing_data/s3_f_side_single_x0y300.png',
    #                                     iterations=3
    #                                     )  # type: ImageProvider
    logger.debug('Using {} as ImageProvider.'.format(type(image_provider).__name__))
    detector = OpenPoseDetector(prototxt_path, caffemodel_path)  # type: PeopleDetector
    matcher = HistogramMatcher()  # type: PersonMatcher
    triangulation = CameraDistanceTriangulation(AVERAGE_PERSON_WAIST_TO_NECK_LENGTH, z_level)  # type: Triangulation
    tracker = HistogramTracker()  # type: PersonTracker
    # endregion

    for i, image_set in enumerate(image_provider):
        logger.info('step {}'.format(i))
        front_image, side_image = image_set

        logger.info('detecting people')
        front_views = detector.detect(front_image, cameras[0])
        side_views = detector.detect(side_image, cameras[1])

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

        # remove unnecessary items, so results queue does not run out of memory
        for person in tracker.people:
            # TODO distance_planes contains lambdas -> can't pickle to queue; but they're just for debug -> remove
            person.time_frames[-1].distance_planes = None  # cleaning the newest time-frame each step

            # views contain images (not needed anymore), leading to results queue memory error -> clean images
            try:
                # last TF needed for tracking of the next TF > do NOT clean the last TF, just the second last
                for view in person.time_frames[-2].views:
                    view.original_image = None
                    view.person_image = None
            except IndexError:
                pass  # person has just one time-frame

        output_queue.put(tracker.people)


def main() -> ExitCode:
    logging.captureWarnings(True)
    warnings.simplefilter('always', ResourceWarning)
    coloredlogs.install(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    logger.debug('main started')

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
    visualizer = Plotter3D([], [camera_front, camera_side])  # type: Visualizer
    results_queue = multiprocessing.Queue()

    processing_thread = multiprocessing.Process(target=processing_pipeline,
                                                args=([camera_front, camera_side], z_level, results_queue))
    processing_thread.start()

    while True:
        while not results_queue.empty():
            visualizer.render(results_queue.get())
            plt.show()

        plt.pause(1)  # evil pause pauses all threads, not just main -> threading won't work, needs multiprocessing
        if cv2.waitKey(2) & 0xFF == ord('q'):
            logger.debug('break')
            break

    # FIXME: Better thread/process handling. Better application shutdown.
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
