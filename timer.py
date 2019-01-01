#!/usr/bin/env python3
"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import timeit

import cv2

import detector
import openpose


def povapose_single_image_single_person():
    """
    Measure pedestrian detection using PovaPose for single image with single person.
    """
    person_detector = openpose.PovaPose.PovaPose(
        prototxt_path="openpose/pose/coco/pose_deploy_linevec.prototxt",
        caffemodel_path="openpose/pose/coco/pose_iter_440000.caffemodel"
    )
    image = cv2.imread('testing_data/s2_f_x0y300.png')

    detection = ('person_detector.set_image_for_detection(image)\n'
                 'people = person_detector.run_multi_person_detection()\n')
    detection_time = timeit.repeat(detection,
                                   globals={'person_detector': person_detector, 'image': image},
                                   number=1,
                                   repeat=3)
    print("PovaPose single_person t={}".format(detection_time))


def openpose_gpu_binary(openpose_binary_path=None, repeat=3):
    """
    Measure pedestrian detection using OpenPose binary for GPU.
    """
    assert openpose_binary_path, 'Provide path to OpenPose binary!'

    # test single image
    image = cv2.imread('testing_data/s2_f_x0y300.png')

    person_detector = detector.OpenPoseBinaryDetector(openpose_binary_path, using_gpu=True)
    detection = 'people = person_detector.detect(image, camera=None)\n'
    detection_time = timeit.repeat(detection,
                                   globals={'person_detector': person_detector, 'image': image},
                                   number=1, repeat=repeat)
    print("{} single image t={}".format(openpose_gpu_binary.__name__, detection_time))

    # test multiple images
    image_count = 10
    images = [image] * image_count
    cameras = [None] * image_count
    detection = 'people = person_detector.detect_multiple_images(images, cameras)\n'
    detection_time = timeit.repeat(detection,
                                   globals={'person_detector': person_detector, 'images': images, 'cameras': cameras},
                                   number=1, repeat=repeat)
    print("{} {} images t={}".format(openpose_gpu_binary.__name__, image_count, detection_time))


def main():
    povapose_single_image_single_person()
    # openpose_gpu_binary(r'')


if __name__ == '__main__':
    main()
