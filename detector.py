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
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import cv2

import openpose
from camera import Camera
from person import PersonView
from utils.enumeration import Enumeration

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


class OpenPoseBinaryDetector(PeopleDetector):
    """
    Detection using pre-compiled binary of OpenPose. Releases can be found at:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases
    """
    class Models(Enumeration):
        coco = 'COCO'
        body_25 = 'BODY_25'

    models = Models()  # enum for OP models

    def __init__(self, binary_path, using_gpu, net_resolution='-1x320', force_op_model=None):
        """
        1) Go to OpenPose releases: https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases
        2) Download and extract OpenPose folder (referred as OP_HOME).
        3) Run OP_HOME/models/getModels.bat to download all OP models.
            [OPTIONAL] edit the getModels.bat to download only needed models.
        4) [OPTIONAL] Read https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/quick_start.md
        :param binary_path: path to OpenPose binary
        :param using_gpu: True to use GPU, False to use CPU. To use GTX GPU, it should support OpenCL 1.2. Install CUDA
            drivers from: https://developer.nvidia.com/cuda-downloads
            Then follow cuDNN installation guide: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
        :param net_resolution: Multiples of 16, e.g. 320x176. Increase ~ accuracy increase. Decrease ~ speed increase.
            For best results, keep the closest aspect ratio possible to the images processed. Using -1 in any of the
            dimensions, OP will choose the optimal aspect ratio depending on the input. E.g. the default -1x368 is
            equivalent to 656x368 for 16:9 resolutions (full HD 1980x1080, HD 1280x720 etc.). NOTE: Higher resolution
            means higher memory consumption (RAM or GPU memory). E.g. -1x336 (for full HD) takes 2 GB of GTX 760 memory.
        :param force_op_model: Manually select OpenPose model BODY_25/COCO. By default, OP model is BODY_25 for GPU,
        COCO for CPU. Because COCO is ~3x faster on CPU than BODY_25, but BODY_25 is ~40% faster on GPU.
            Make sure to have the requested model downloaded in OP_HOME/models folder.
        """
        if 'Windows' not in platform.system():
            raise NotImplementedError("Only Windows binaries supported.")

        assert os.path.isfile(binary_path), "OpenPose binary not found. Path: {}".format(binary_path)
        self.binary_home = os.path.dirname(binary_path).rstrip('bin')  # OpenPoseDemo.exe is in bin/ subdirectory

        if force_op_model and not self.models.contains(force_op_model):
            logger.error("Using an unknown OpenPose model: {0}. Select from {1}.models: {2}!"
                         .format(force_op_model, type(self).__name__,  self.models.enum_values()))
            raise NotImplementedError("Using unknown OpenPose model!")

        # prepare tmp directory for input images and results; tmp dir is deleted in obj destructor
        tmp_dir_name = 'POVa_pedestrian_tracking_TEMP_DIR'
        try:
            self.tmp_dir = os.path.join(os.environ['TEMP'], tmp_dir_name)
        except KeyError:
            logger.warning("Environment variable TEMP not found. Creating tmp folder in CWD.")
            self.tmp_dir = os.path.join(os.getcwd(), tmp_dir_name)

        logger.debug('Detector created tmp dir at: {}'.format(self.tmp_dir))
        self.images_folder = os.path.join(self.tmp_dir, 'images')
        self.results_folder = os.path.join(self.tmp_dir, 'results')
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)

        # create a command to run binary file
        # out_res and render_pose disabled intentionally (even if no GUI at all), so they are not calculated -> faster
        self.cmd = binary_path
        self.cmd += ' --image_dir {}'.format(self.images_folder)  # load images for detection from here
        self.cmd += ' --write_json {}'.format(self.results_folder)  # output JSONs are saved here
        self.cmd += ' --net_resolution {}'.format(net_resolution)  # lower means faster, but less accurate
        self.cmd += ' --display 0'  # disable GUI mode -> speedup
        self.cmd += ' --output_resolution 0x0'  # don't display the image -> speedup
        self.cmd += ' --render_pose 0'  # don't draw result into the image -> speedup
        if using_gpu is True:
            self.model = self.models.body_25
            self.cmd += ' --num_gpu 1' if using_gpu else ''  # use one GPU; no auto-detection -> faster
        else:
            self.model = self.models.coco

        if force_op_model:
            self.model = force_op_model

        self.cmd += ' --model_pose {}'.format(self.model)

    def detect(self, image, camera: Camera) -> List[PersonView]:
        """
        Detect people in one image. For multiple images use method <detect_multiple_images>. It is much faster, because
        OP initialization for every single image takes time. OP initialization ~ 1-2 sec, but detection of 1 image
        on GPU ~ 0.1-0.5 sec. E.g. detect 10 images takes 10*2+10*0.5 = 25 sec. However, detect 10 images using
        the <detect_multiple_images> method takes 1*2+10*0.5 = 7 sec.
        """
        # prepare the image for detection
        img_name = 'image.png'
        result_name = 'image_keypoints.json'
        # OpenPose binary reads images from a given directory, so we need to write images to the directory first
        cv2.imwrite(os.path.join(self.images_folder, img_name), image)

        # run detection
        p = subprocess.Popen(self.cmd, cwd=self.binary_home, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cmd_result = p.communicate()
        if p.returncode != 0:
            logger.error("OpenPose binary run failed!\nSTDOUT: {}\nSTDERR: {}".format(cmd_result[0], cmd_result[1]))
            raise RuntimeError("OpenPose binary run failed!")
        else:
            logger.debug("OpenPose binary run success. STDOUT: {}".format(cmd_result[0]))

        # parse detection results to person views
        views = self.load_valid_persons_from_json(os.path.join(self.results_folder, result_name), image, camera)

        return views

    def detect_multiple_images(self, images, cameras: List[Camera]) -> List[List[PersonView]]:
        """
        Detection of multiple images at once is MUCH FASTER, because OP is initialized just once for multiple
        images. Initialization ~ 1-2 sec; detection of 1 image on GPU ~ 0.1-0.5 sec. NOTE: Memory consumption is
        determined by the net_resolution, not by the number of images to detect (images are processed one-by-one).
        """
        results = []

        # prepare images for detection
        for i, image in enumerate(images):
            img_name = 'image{}.png'.format(i)
            # OpenPose binary reads images from a given directory, so we need to write images to the directory first
            cv2.imwrite(os.path.join(self.images_folder, img_name), image)

        # run detection
        p = subprocess.Popen(self.cmd, cwd=self.binary_home, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cmd_result = p.communicate()
        if p.returncode != 0:
            logger.error("OpenPose binary run failed!\nSTDOUT: {}\nSTDERR: {}".format(cmd_result[0], cmd_result[1]))
            raise RuntimeError("OpenPose binary run failed!")
        else:
            logger.debug("OpenPose binary run success. STDOUT: {}".format(cmd_result[0]))

        # parse detection results to person views
        for i, image in enumerate(images):
            result_name = 'image{}_keypoints.json'.format(i)
            views = self.load_valid_persons_from_json(os.path.join(self.results_folder, result_name), image, cameras[i])
            results.append(views)

        return results

    def load_valid_persons_from_json(self, json_path, image, camera: Camera) -> List[PersonView]:
        with open(json_path) as json_file:
            detection_result = json.load(json_file)

        results = []
        for person in detection_result['people']:
            body_parts = self.get_body_parts_from_keypoints(person['pose_keypoints_2d'])
            person_image = self.get_person_subimage(image, body_parts)
            neck, hip_center = self.get_neck_and_hip_coordinates(body_parts)
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
        i = 2  # starting from the first confidence index
        for detection_confidence in keypoints[2::3]:
            # keypoint is defined as (part X, part Y, probability)
            if detection_confidence >= part_confidence_threshold:
                body_parts.append((int(keypoints[i-2]), int(keypoints[i-1])))
            else:
                body_parts.append(None)

            i += 3  # iterating by 3

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

    def get_neck_and_hip_coordinates(self, body_parts: List[Optional[Tuple[int, int]]]):
        """
        Get coordinates of neck and center of hips. If only one hip detected, return that hip (instead of the center).
        If body part not detected, return None for that part. Keypoints (body parts) order can be found at:
        https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/v1.4.0/doc/output.md
        """
        if self.model == self.models.body_25:
            neck = body_parts[1]
            hip = body_parts[8]
            hip_r = body_parts[9]
            hip_l = body_parts[12]
        elif self.model == self.models.coco:
            neck = body_parts[1]
            hip = None
            hip_r = body_parts[9]
            hip_l = body_parts[12]
        else:
            raise NotImplementedError("Unknown OpenPose model!")

        return neck, self._get_optimal_hip_coordinate(hip, hip_r, hip_l)

    @staticmethod
    def _get_optimal_hip_coordinate(hip, hip_l, hip_r) -> Optional[Tuple[int, int]]:
        """Get center of hips. If only one of the hips detected, return that hip (instead of the center)."""
        if hip:
            return hip
        elif hip_r and hip_l:
            return int((hip_r[0] + hip_l[0]) / 2), int((hip_r[1] + hip_l[1]) / 2)
        elif hip_r:
            return hip_r
        elif hip_l:
            return hip_l
        else:
            return None

    def __del__(self):
        """Delete detector's temporary folder, so images and results are not kept for another run."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)  # ignore e.g. folder doesn't exist (if deleted manually)
