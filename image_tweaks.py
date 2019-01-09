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

import cv2
import numpy as np

from utils.image_processing import synchronize_image
from tools import select_rectangle_mask_using_mouse

logger = logging.getLogger(__name__)


class ImageTweaks(ABC):
    """
    TODO name? image processing/processor, image tweaks/tweaker, image preparation/preparator?
    This class is used to process images. It can be used by image_provider to adjust images by needs.
    """
    @abstractmethod
    def apply(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        TODO
        GaussianBlur(),medianBlur(),bilateralFilter()
        Smooth the image with one of the linear or non-linear filters

        Sobel(), Scharr() Compute the spatial image derivatives

        erode(), dilate() Morphological operations
        """
        pass


class ImgTweaksBasedOnPresumablySameArea(ImageTweaks):
    """
    Images processing is based on a real-world area that should look the same in all the images. E.g. a wall should be
    the same light blue in all images. To do this, manually select (interactive mode) or provide masks, so the mask in
    each image is masking the target area (e.g. the wall).
    The images are first to synchronized to a reference image corresponding with their source (camera/video).
    Then
    NOTE: The masked area should not get covered (e.g. people
    walking in front of it).
    """

    def __init__(self, interactive=True, masks=None, blur_images=False):
        """
        :param interactive: whether to select masks manually (using mouse in displayed images)
        :param masks: number of image masks MUST match number of images that will be processed later
        :param blur_images: blur images little bit, so histograms may (or may not) be a bit more stable
        """
        if interactive:
            self.interactive = True
            self.masks = None  # list of masks used to mark the target area in images will be selected at the first use
        else:
            self.interactive = False
            self.masks = masks

        self.reference_images = None  # list of reference images for each source is created at the first use
        self.is_first_run = True
        self.blur_images = blur_images

    def apply(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Images are blurred a little bit.
        Images from the first step are taken as reference images for each source (index).
        Images from one source are synced with a reference image from that source.
        Images in one step/iteration are then synced with one image from that step.
        """
        # TODO add blur? compare results with/without blur and decide
        if self.is_first_run:
            self.init_based_on_input(images)

        # Synchronize lightness and colors:
        # Images from one source are synced with a reference image from that source.
        images = self.sync_images_with_their_source_reference(images)
        # Images in one step/iteration are then synced with one image from that step.
        images = self.sync_images_mutually(images)

        if self.blur_images:
            for i, image in enumerate(images):
                cv2.blur(image, (3, 3), images[i])

        return images

    def init_based_on_input(self, images):
        self.is_first_run = False
        self.reference_images = images
        if self.masks is None:
            if self.interactive:
                self.select_masks_for_img_processing(images)
            else:
                logger.warning("{}: no masks given, using the whole images as reference.".format(type(self).__name__))
                self.masks = [None] * len(images)  # set mask for all images to None > masks are not used

    def select_masks_for_img_processing(self, images):
        """
        Select masks to later sync images with their reference img - the first img from that source (index). Should be
        an area that won't get covered by anything and is stable. Should cover the same real-world area in all images.
        """
        window_name = '{} - mask selection for images sync'.format(type(self).__name__)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self.masks = list(map(lambda img: select_rectangle_mask_using_mouse(window_name, img), images))
        cv2.destroyWindow(window_name)

    def sync_images_with_their_source_reference(self, images):
        """Synchronize lightness and colors of each image with its reference image (based on source/index)."""
        synced_images = []
        # sync every image with its corresponding reference image
        for i, image in enumerate(images):
            synced_image = synchronize_image(self.reference_images[i], image, self.masks[i], self.masks[i])
            synced_images.append(synced_image)

        return synced_images

    def sync_images_mutually(self, images):
        """Synchronize lightness and colors of images mutually - sync them with the first one in the list."""
        synced_images = []
        first_source_image = None  # other images in current step are synced with this one
        for i, image in enumerate(images):
            if i == 0:
                first_source_image = image
                synced_image = image
            else:
                synced_image = synchronize_image(first_source_image, image, self.masks[0], self.masks[i])

            synced_images.append(synced_image)

        return synced_images
