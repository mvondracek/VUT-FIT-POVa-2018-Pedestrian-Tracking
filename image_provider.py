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

from image_tweaks import ImageTweaks

logger = logging.getLogger(__name__)


class ImageProvider(ABC):
    """Iterator provides tuples of images, 1 from each camera. Tuples should be provided in a chronological order. """
    @abstractmethod
    def __iter__(self):
        """ Return the iterator object - most probably self. """
        pass

    @abstractmethod
    def __next__(self) -> Tuple:
        """ Load next step images. Must raise StopIteration if no more items. """
        pass


class DummyImageProvider(ImageProvider):
    """Dummy provider provides a predefined pair of images for a number of iterations."""
    def __init__(self, front_image_path, side_image_path, iterations: int = 1):
        self.front_image = cv2.imread(front_image_path)
        self.side_image = cv2.imread(side_image_path)
        self.iterations = iterations

    def __iter__(self):
        return self

    def __next__(self) -> Tuple:
        if self.iterations > 0:
            self.iterations -= 1
            return self.front_image, self.side_image
        else:
            raise StopIteration


class ImageProviderFromVideo(ImageProvider):
    """ Load images from given video(s). """
    def __init__(self, paths_to_videos, start: int = 0, skipping: int = 0, image_tweaker: ImageTweaks = None):
        """"
        :param start: Start providing images from frame with this ordinary number. Skip first `start` frames.
        :param skipping: Skip specified number of frames each time before providing next image.
        :param image_tweaker: optional image processing that is applied to images
        """
        self.finished = False
        self.image_tweaks = image_tweaker
        self.videos = []
        for path in paths_to_videos:
            video = cv2.VideoCapture(path)
            self.videos.append(video)
            if not video.isOpened():
                raise IOError("Image provider failed to open a video. Path: {}".format(path))

        self.skipping = skipping
        for video in self.videos:
            video.set(cv2.CAP_PROP_POS_FRAMES, start)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple:
        if self.finished:
            raise StopIteration

        images = []
        for video in self.videos:
            position = video.get(cv2.CAP_PROP_POS_FRAMES)
            video.set(cv2.CAP_PROP_POS_FRAMES, position + self.skipping)
            ret, frame = video.read()
            if not ret:
                logger.debug("Video stream has ended.")
                self.finished = True
                self._release_videos()
                raise StopIteration
            else:
                images.append(frame)

        if self.image_tweaks:
            images = self.image_tweaks.apply(images)

        if __debug__:
            self.show_images(images)

        return tuple(images)

    @staticmethod
    def show_images(images):
        for i, image in enumerate(images):
            window_name = 'ImageProviderFromVideo video={}'.format(i)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image)
            cv2.resizeWindow(window_name, 384, 216)
            cv2.waitKey(1)

    def _release_videos(self):
        """OpenCV video readers' resources should be released properly."""
        map(lambda video: video.release(), self.videos)

