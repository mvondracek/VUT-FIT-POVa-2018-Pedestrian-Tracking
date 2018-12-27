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

import openpose


def povapose_single_person():
    """
    Measure pedestrian detection using PovaPose for single image with single person.
    """
    person_detector = openpose.PovaPose.PovaPose(
        prototxt_path="../openpose/pose/coco/pose_deploy_linevec.prototxt",
        caffemodel_path="../openpose/pose/coco/pose_iter_440000.caffemodel"
    )
    image = cv2.imread('../testing_data/s2_f_x0y300.png')

    detection = ('person_detector.set_image_for_detection(image)\n'
                 'people = person_detector.run_multi_person_detection()\n')
    detection_time = timeit.repeat(detection,
                                   globals={'person_detector': person_detector, 'image': image},
                                   number=1,
                                   repeat=3)
    print("PovaPose single_person t={}".format(detection_time))


def main():
    povapose_single_person()


if __name__ == '__main__':
    main()
