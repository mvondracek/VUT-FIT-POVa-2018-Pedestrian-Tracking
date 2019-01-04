"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
from typing import Tuple

import numpy as np


class Camera:
    def __init__(self, name: str, focal_length: float, position: Tuple[int, int, int],
                 orientation: Tuple[int, int, int]):
        """
        :param name: camera name used in user interface
        :param focal_length: focal length of camera, see `Camera.calibrate_focal_length`
        :param position: 3D coordinates of the camera position
        :param orientation: 3D vector of the camera's view orientation
        """
        self.name = name
        self.focal_length = focal_length
        self.position = position
        self.orientation = orientation / np.linalg.norm(orientation)  # convert to unit vector

    @staticmethod
    def calibrate_focal_length(real_distance: int, real_size: int, pixel_size: int) -> float:
        """
        Calibrate focal length of the camera based on measurement of known object and its image representation.
        :param real_distance: real distance of known object from camera in millimeters
        :param real_size: real size size of known object in millimeters,
        :param pixel_size: size of known object measured in pixels in image obtained from the camera
        :return: focal length
        """
        return (pixel_size * real_distance) / real_size
