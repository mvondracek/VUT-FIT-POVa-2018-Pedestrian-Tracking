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
    def __init__(self, front_video_path, side_video_path):
        logger.debug('Using ImageProviderFromVideo as ImageProvider.')
        self.front_video = cv2.VideoCapture(front_video_path)
        self.side_video = cv2.VideoCapture(side_video_path)

    def get_next_images(self) -> Tuple:
        ret1, front = self.front_video.read()
        ret2, side = self.side_video.read()
        if ret1 is None or ret2 is None:
            logger.info("Video stream ended.")
            return None, None
        else:
            return front, side
