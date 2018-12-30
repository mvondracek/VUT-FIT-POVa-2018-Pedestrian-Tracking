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
from typing import List, Tuple, Optional, Union

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
    """Run OpenPose detection using pre-built binary."""
    def __init__(self, binary_path, use_gpu, net_resolution='-1x368', force_op_model=None):
        """
        1) Go to OpenPose releases: https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases
        2) Download and extract OpenPose folder (referred as OP_HOME).
        3) Run OP_HOME/models/getModels.bat to download all OP models.
            [OPTIONAL] edit the getModels.bat to download only needed models.
        4) [OPTIONAL] Read https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/quick_start.md
        :param binary_path: path to OpenPose binary
        :param use_gpu: True to use GPU, False to use CPU. By default, OP model is BODY_25 for GPU, COCO for CPU. Make
            sure to have the corresponding model downloaded in OP_HOME/models folder, or select other model.
        :param net_resolution: Multiples of 16, e.g. 320x176. Increase ~ accuracy increase. Decrease ~ speed increase.
            For best results, keep the closest aspect ratio possible to the images processed. Using -1 in any of the
            dimensions, OP will choose the optimal aspect ratio depending on the input. E.g. the default -1x368 is
            equivalent to 656x368 for 16:9 resolutions (full HD 1980x1080, HD 1280x720 etc.).
        :param force_op_model: Manually select OpenPose model BODY_25/COCO/MPI. By default, the best suitable OP model
            is chosen for binary type. E.g. COCO is ~3x faster on CPU than BODY_25, but BODY_25 is ~40% faster on GPU.
            Make sure to have the requested model downloaded in OP_HOME/models folder.
        """
        # TODO add info about running CUDA and cuDNN (links)
        # TODO reflect model selection and cpu/gpu usage
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
        img_name = 'image.png'
        result_name = 'image_keypoints.json'

        # prepare the image for detection
        cv2.imwrite(os.path.join(self.images_folder, img_name), image)

        # run detection
        p = subprocess.Popen(self.cmd, cwd=self.binary_home)
        result = p.communicate()  # TODO detekovat errory, nevypisovat na stdout

        # parse detection results to person views
        views = self.load_all_valid_persons_from_json(os.path.join(self.results_folder, result_name), image, camera)

        return views

    def detect_multiple_images(self):
        raise NotImplementedError  # TODO

    def load_all_valid_persons_from_json(self, json_path, image, camera: Camera) -> List[PersonView]:
        with open(json_path) as json_file:
            detection_result = json.load(json_file)

        results = []
        for person in detection_result['people']:
            body_parts = self.get_body_parts_from_keypoints(person['pose_keypoints_2d'])
            person_image = self.get_person_subimage(image, body_parts)
            neck, hip_center = self.get_neck_and_hips_center_coordinates(body_parts)
            if not neck or not hip_center:
                logger.warning("Person does not have nose or hips detected.")
                continue

            results.append(PersonView(image, person_image, camera, neck, hip_center))

        return results

    @staticmethod
    def get_body_parts_from_keypoints(keypoints: List[float], part_confidence_threshold=0.7) -> List[Optional[Tuple[int, int]]]:
        """
        Body parts are loaded only if detection confidence (0 to 1) is higher than the part confidence threshold.
        :return: body parts coordinates (x, y); order of parts can be found at:
        https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering
        """
        body_parts = []
        for i, detection_confidence in enumerate(keypoints[2::3]):
            # keypoint is defined as (part X, part Y, probability)
            if detection_confidence >= part_confidence_threshold:
                body_parts.append((int(keypoints[i-2]), int(keypoints[i-1])))
            else:
                body_parts.append(None)

        return body_parts

    @staticmethod
    def get_person_subimage(image, body_parts: List[Optional[Tuple[int, int]]]):
        """Return subimage defined by person's bounding box."""
        # TODO combine this with background substraction to extract just the person, not background
        top_left_x = min(part[0] for part in body_parts if part is not None)
        top_left_y = min(part[1] for part in body_parts if part is not None)
        bottom_right_x = max(part[0] for part in body_parts if part is not None)
        bottom_right_y = max(part[1] for part in body_parts if part is not None)

        # last item in indexed range is excluded in python, but we want the bottom_right point included -> need +1
        return image[top_left_y:bottom_right_y+1, top_left_x:bottom_right_x+1]

    @staticmethod
    def get_neck_and_hips_center_coordinates(body_parts: List[Optional[Tuple[int, int]]]):
        """
        Get coordinates of neck and center of hips. If only one hip detected, return that hip (instead of the center).
        Keypoints (body parts) order can be found at:
        https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering
        """
        neck = body_parts[1]
        hip = None
        hip_r = body_parts[9]
        hip_l = body_parts[12]

        if hip_r and hip_l:
            hip = (int((hip_r[0] + hip_l[0]) / 2), int((hip_r[1] + hip_l[1]) / 2))
        elif hip_r:
            hip = hip_r
        elif hip_l:
            hip = hip_l

        return neck, hip
