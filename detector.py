"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import json
import logging
import platform
import os
import subprocess
from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2

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
        self.binary_home = os.path.dirname(binary_path).rstrip('bin')  # OpenPoseDemo.exe is in bin/ subdirectory

        # prepare tmp directory for input images and results
        try:
            tmp_dir = os.path.join(os.environ['TEMP'], 'POVa_pedestrian_tracking')
        except KeyError:
            logger.warning("Environment variable TEMP not found. Creating tmp folder in CWD.")
            tmp_dir = os.path.join(os.getcwd(), 'POVa_pedestrian_tracking')

        logger.debug('Detector created tmp dir at: {}'.format(tmp_dir))
        self.images_folder = os.path.join(tmp_dir, 'images')
        self.results_folder = os.path.join(tmp_dir, 'results')
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)

        # create a command to run binary file
        self.cmd = binary_path
        self.cmd += ' --image_dir {}'.format(self.images_folder)  # load images for detection from here
        self.cmd += ' --write_json {}'.format(self.results_folder)  # output JSONs are saved here
        self.cmd += ' --net_resolution {}'.format(net_resolution)  # lower means faster, but less accurate
        self.cmd += ' --num_gpu 1' if use_gpu else ''  # use only one GPU
        self.cmd += ' --display 0'  # disable GUI mode -> speedup
        self.cmd += ' --output_resolution 0x0'  # don't display the image -> speedup
        self.cmd += ' --render_pose 0'  # don't draw result into the image -> speedup
        # out_res and render_pose disabled intentionally (even if no GUI at all), so it is not calculated -> faster
        # TODO add ' --model_pose {}' ... BODY_25 for GPU or COCO for CPU

    def detect(self, image, camera: Camera) -> List[PersonView]:
        # TODO predelat parametry na dict Cam->Img (poradi?) nebo na list Tuplu (image, camera)???
        img_name = 'image.png'
        result_name = 'image_keypoints.json'
        cv2.imwrite(os.path.join(self.images_folder, img_name), image)

        p = subprocess.Popen(self.cmd, cwd=self.binary_home)
        result = p.communicate()  # TODO detekovat errory, nevypisovat na stdout

        people = self._get_people_from_json(os.path.join(self.results_folder, result_name))

        detected = []
        for person in people:
            detected.append(PersonView(image, person[0], camera, (person[1][0], person[1][1]), (person[2][0], person[2][1])))
        return detected

    def _get_people_from_json(self, path):  # TODO rename
        with open(path) as f:
            detection_result = json.load(f)

        people_parts = []
        for person in detection_result['people']:
            people_parts.append(self._get_body_parts_from_keypoints(person['pose_keypoints_2d']))

    @staticmethod
    def _get_body_parts_from_keypoints(keypoints: List[float], part_confidence_threshold=0.7) -> List[Tuple[float, float] or None]:
        """Body parts are loaded only if detection confidence (0 to 1) is higher than the part confidence threshold."""
        body_parts = []
        for i, detection_confidence in enumerate(keypoints[2::3]):
            # keypoint is defined as (part X, part Y, probability)
            if detection_confidence >= part_confidence_threshold:
                body_parts.append((keypoints[i-2], keypoints[i-1]))
            else:
                body_parts.append(None)

        return body_parts


o = OpenPoseDetectorUsingPrecompiledBinary(r'C:\Users\Filip\Downloads\openpose-1.4.0-win64-gpu-binaries\bin\OpenPoseDemo.exe')
# img = cv2.imread('testing_data/s3_f_side_multi_y600.png')
img = cv2.imread('testing_data/s3_m_front_.png')
o.detect(img, Camera('asd', 1.0, (1, 0, 2), (5, 3, 0)))


"""Error:
Prototxt file not found: models\pose/body_25/pose_deploy.prototxt.
	3. Using paths with spaces.
"""
