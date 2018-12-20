"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import logging
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
        logger.info("Detected people = {0}".format(len(people)))
        detected = []
        for person in people:
            detected.append(PersonView(person[0], camera, (person[1][0], person[1][1]), (person[2][0], person[2][1])))
        return detected
