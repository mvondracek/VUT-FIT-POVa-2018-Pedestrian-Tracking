"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import logging
import platform
import os
from abc import ABC, abstractmethod
from typing import List

import openpose
from camera import Camera
from person import PersonView

logger = logging.getLogger(__name__)


class PeopleDetector(ABC):
    @abstractmethod
    def detect(self, image, camera: Camera) -> List[PersonView]:
        """
        Detect people in given image.
        :param image: image from camera for pedestrian detection
        :param camera: camera which taken current image
        :return: Views of detected people including their pose
        """
        pass


class OpenPoseDetector(PeopleDetector):
    """
    People detection using `OpenPoseDetector <https://github.com/CMU-Perceptual-Computing-Lab/openpose>`_
    """

    def __init__(self, prototxt_path, caffemodel_path):
        logger.debug('Using OpenPoseDetector as PeopleDetector.')
        self.pova_pose = openpose.PovaPose.PovaPose(prototxt_path, caffemodel_path)

    def detect(self, image, camera: Camera) -> List[PersonView]:
        self.pova_pose.set_image_for_detection(image)
        people = self.pova_pose.run_multi_person_detection()
        logger.debug("Camera {}, detected people = {}".format(camera.name, len(people)))
        detected = []
        for person in people:
            detected.append(PersonView(image, person[0], camera, (person[1][0], person[1][1]), (person[2][0], person[2][1])))
        return detected


class OpenPoseDetectorUsingPrecompiledBinary(PeopleDetector):
    """TODO"""
    def __init__(self, binary_path, use_gpu=True, net_resolution='480x240'):
        if 'Windows' not in platform.system():
            raise NotImplementedError("Only Windows binaries supported.")

        assert os.path.isfile(binary_path), 'OpenPose binary not found. Path: {}'.format(binary_path)
        try:
            images_folder = os.path.join(os.environ['TEMP'], 'POVa_pedestrian_tracking', 'images')
            results_folder = os.path.join(os.environ['TEMP'], 'POVa_pedestrian_tracking', 'results')
        except KeyError:
            logger.warning("Environment variable TEMP not found. Creating tmp folder in CWD.")
            images_folder = os.path.join(os.getcwd(), 'POVa_pedestrian_tracking', 'images')
            results_folder = os.path.join(os.getcwd(), 'POVa_pedestrian_tracking', 'results')

        self.cmd = binary_path
        self.cmd += ' --image_dir {}'.format(images_folder)  # load images for detection from here
        self.cmd += ' --write_json {}'.format(results_folder)  # output JSONs are saved here
        self.cmd += ' --net_resolution {}'.format(net_resolution)  # lower means faster, but less accurate
        self.cmd += ' --num_gpu 1' if use_gpu else ''  # use only one GPU
        self.cmd += ' --display 0'  # disable GUI mode -> speedup
        self.cmd += ' --output_resolution 0x0'  # don't display the image -> speedup
        self.cmd += ' --render_pose 0'  # don't draw result into the image -> speedup
        # out_res and render_pose disabled intentionally (even if no GUI at all), so it is not calculated -> faster

    def detect(self, image, camera: Camera) -> List[PersonView]:
        pass
