import cv2
import numpy as np
from tools import select_rectangle_mask_using_mouse


def synchronize_image(ref_img, image, mask1=None, mask2=None, interactive=False):
    """
    Synchronize image with ref_image based on mean color & light value in the whole images or in areas specified by masks.
    Mask area can also be selected manually using mouse in interactive mode.
    :param ref_img: sample image (BGR type), remains unchanged
    :param image: image (BGR type) that is adjusted to correspond with light intensity of ref_img
    :param mask1: mean value of light is calculated only in this area for ref_img
    :param mask2: mean value of light is calculated only in this area for image
    :param interactive: True to enable interactive mode where mask can be selected manually in image view
    :return: image2 (changed)
    """
    # windows used only if interactive selection
    window1 = None
    window2 = None
    if interactive:
        window1 = 'Sync Images - ref_img: interactive mask selection'
        window2 = 'Sync Images - image: interactive mask selection'
        mask1 = select_rectangle_mask_using_mouse(window1, ref_img)
        mask2 = select_rectangle_mask_using_mouse(window2, image)

    # adjust image (based on ref_img), ref_img is untouched
    image = synchronize_image_colors(ref_img, image, mask1=mask1, mask2=mask2, interactive=False)
    image = synchronize_image_light_intensity(ref_img, image, mask1=mask1, mask2=mask2, interactive=False)

    # show results in interactive mode (to see the changes)
    if interactive:
        cv2.imshow(window1, ref_img)
        cv2.imshow(window2, image)
        cv2.waitKey()
        cv2.destroyWindow(window1)
        cv2.destroyWindow(window2)

    return image


def synchronize_image_colors(image1, image2, mask1=None, mask2=None, interactive=False):
    """
    Synchronize image2 with image1 based on mean color values in the whole images or in areas specified by masks.
    Mask area can also be selected manually using mouse in interactive mode. Image1 remains unchanged.
    """
    # windows used only if interactive selection
    window1 = None
    window2 = None
    if interactive:
        window1 = 'Sync Images Colors - image 1: interactive mask selection'
        window2 = 'Sync Images Colors - image 2: interactive mask selection'
        mask1 = select_rectangle_mask_using_mouse(window1, image1)
        mask2 = select_rectangle_mask_using_mouse(window2, image2)

    # count mean values of color channels and their ratio between image1 and image2
    means1, _ = cv2.meanStdDev(image1, mask=mask1)
    means2, _ = cv2.meanStdDev(image2, mask=mask2)
    ratios = means1 / means2
    ratios = ratios.flatten()  # shape (3,1) -> shape (3); just 1 non-nested value for each color channel

    # adjust image2, so it should be more similar to image1
    image1 = image1.astype(np.float16)
    image2 = image2.astype(np.float16)
    image2[:, :] = image2[:, :] * ratios
    np.clip(image2, a_min=0, a_max=255, out=image2)
    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)

    # show results in interactive mode (to see the changes)
    if interactive:
        cv2.imshow(window1, image1)
        cv2.imshow(window2, image2)
        cv2.waitKey()
        cv2.destroyWindow(window1)
        cv2.destroyWindow(window2)

    return image2


def synchronize_image_light_intensity(image1, image2, mask1=None, mask2=None, interactive=False):
    """
    Synchronize image2 with image1 based on mean light value in the whole images or in areas specified by masks.
    Mask area can also be selected manually using mouse in interactive mode. Image1 remains unchanged.
    """
    # windows used only if interactive selection
    window1 = None
    window2 = None
    if interactive:
        window1 = 'Sync Images Lightness - image 1: interactive mask selection'
        window2 = 'Sync Images Lightness - image 2: interactive mask selection'
        mask1 = select_rectangle_mask_using_mouse(window1, image1)
        mask2 = select_rectangle_mask_using_mouse(window2, image2)

    # convert images to LAB format: L = Lightness (intensity); A = color component (Green to Magenta); B – color component (Blue to Yellow)
    light_channel = 0
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

    # get ratio of lightness between the images
    mean1, _ = cv2.meanStdDev(image1[:, :, light_channel], mask=mask1)
    mean2, _ = cv2.meanStdDev(image2[:, :, light_channel], mask=mask2)
    ratio = mean1 / mean2
    ratio = ratio.flatten()  # just 1 non-nested value

    # adjust image 2, so it matches mean lightness of image 1
    image1 = image1.astype(np.float16)
    image2 = image2.astype(np.float16)
    image2[:, :, light_channel] = image2[:, :, light_channel] * ratio
    np.clip(image2, a_min=0, a_max=255, out=image2)

    # convert images back to BGR
    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_LAB2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_LAB2BGR)

    # show results in interactive mode (to see the changes)
    if interactive:
        cv2.imshow(window1, image1)
        cv2.imshow(window2, image2)
        cv2.waitKey()
        cv2.destroyWindow(window1)
        cv2.destroyWindow(window2)

    return image2
