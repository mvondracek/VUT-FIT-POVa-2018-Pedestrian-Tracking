"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import cv2

logger = logging.getLogger(__name__)


class ImageProvider(ABC):
    """
    ImageProvider should provide pairs of front & side images. Image pairs should be provided in a chronological order.
    """
    @abstractmethod
    def get_next_images(self) -> Tuple:
        """Load next available image pair (front image & side image)."""
        pass


class DummyImageProvider(ImageProvider):
    """Dummy provider provides the same predefined pair of images infinitely."""
    def __init__(self, front_image_path, side_image_path):
        logger.debug('Using DummyImageProvider as ImageProvider.')
        self.front_image = cv2.imread(front_image_path)
        self.side_image = cv2.imread(side_image_path)

    def get_next_images(self) -> Tuple:
        return self.front_image, self.side_image


class ImageProviderFromVideo(ImageProvider):
    def __init__(self, front_video_path, side_video_path, start: int = 0, skipping: int = 0):
        """
        :param start: Start providing images from frame with this ordinary number. Skip first `start` frames.
        :param skipping: Skip specified number of frames each time before providing next image.
        """
        logger.debug('Using ImageProviderFromVideo as ImageProvider.')
        self.front_video = cv2.VideoCapture(front_video_path)
        self.side_video = cv2.VideoCapture(side_video_path)
        self.skipping = skipping
        for i in range(start):
            # TODO: mvondracek: is there a better way how to skip several frames?
            _, front = self.front_video.read()
            _, side = self.side_video.read()
            if __debug__:
                cv2.imshow('ImageProviderFromVideo {} front'.format(id(self)), front)
                cv2.imshow('ImageProviderFromVideo {} side'.format(id(self)), side)
                cv2.waitKey(1)

    def get_next_images(self) -> Tuple:
        ret1, front = None, None
        ret2, side = None, None
        for i in range(self.skipping):
            # TODO: mvondracek: is there a better way how to skip several frames?
            ret1, front = self.front_video.read()
            ret2, side = self.side_video.read()
            if __debug__:
                cv2.imshow('ImageProviderFromVideo {} front'.format(id(self)), front)
                cv2.imshow('ImageProviderFromVideo {} side'.format(id(self)), side)
                cv2.waitKey(1)

        if ret1 is None or ret2 is None:
            logger.info("Video stream ended.")
            return None, None
        else:
            return front, side
