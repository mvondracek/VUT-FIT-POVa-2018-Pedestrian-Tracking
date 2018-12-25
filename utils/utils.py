import math

import cv2
import numpy as np


def synchronize_colors(image1, image2, mask1=None, mask2=None, interactive=False):  # TODO rename
    """#TODO"""
    # windows for interactive selection
    window1 = None
    window2 = None
    if interactive:
        raise NotImplementedError  # TODO implement manual selection of image mask by mouse
        window1 = 'image1_interactive_mask_selection'
        cv2.namedWindow(window1, cv2.WINDOW_NORMAL)
        window2 = 'image2_interactive_mask_selection'
        cv2.namedWindow(window2, cv2.WINDOW_NORMAL)
        cv2.imshow(window1, image1)
        cv2.imshow(window2, image2)
        cv2.waitKey()

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

    if interactive:
        cv2.imshow(window1, image1)
        cv2.imshow(window2, image2)
        cv2.waitKey()

    return image1, image2


synchronize_colors(None, None, None, None, interactive=True)  # TODO

def synchronize_light(image1, image2, mask1=None, mask2=None, interactive=False):  # TODO rename
    """TODO"""
    # TODO zkusit rectangle prumer, nebo prevest do jineho formatu a upravit jas - HSV / LAB
    image1 = cv2.imread('../testing_data/s3_m_front_multi_y600.png')  # TODO Implement image provider.
    image2 = cv2.imread('../testing_data/s3_f_side_multi_y600.png')  # TODO Implement image provider.
    if interactive:
        raise NotImplementedError  # TODO implement manual selection of image mask by mouse
        window1 = 'image1_interactive_mask_selection'
        cv2.namedWindow(window1, cv2.WINDOW_NORMAL)
        window2 = 'image2_interactive_mask_selection'
        cv2.namedWindow(window2, cv2.WINDOW_NORMAL)
        cv2.imshow(window1, image1)
        cv2.imshow(window2, image2)
        cv2.waitKey()

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

    point1 = [(1180, 583), (1212, 597)]
    point2 = [(129, 472), (163, 489)]

    # mean only selected area
    #mean1, _ = cv2.meanStdDev(image1[point1[0][1]:point1[1][1], point1[0][0]:point1[1][0]])
    #mean2, _ = cv2.meanStdDev(image2[point1[0][1]:point1[1][1], point2[0][0]:point1[1][0]])

    # mean whole image
    mean1, _ = cv2.meanStdDev(image1)
    mean2, _ = cv2.meanStdDev(image2)
    #print(mean1)
    #print(mean2)

    image1 = image1.astype(np.float16)
    image2 = image2.astype(np.float16)
    ratio_hue = mean1[0] / mean2[0]
    print(ratio_hue)
    image2[:, :, 0] = image2[:, :, 0] * ratio_hue

    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)

    image1 = cv2.cvtColor(image1, cv2.COLOR_LAB2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_LAB2BGR)
    if interactive:
        cv2.imshow(window1, image1)
        cv2.imshow(window2, image2)
        cv2.waitKey()

    return image1, image2


def calculate_flat_histogram(image):
    """ Calculate flattened normalized histogram (1-D array) of the image. """
    hist = cv2.calcHist([image], channels=[0, 1, 2], mask=None, histSize=[16, 16, 16], ranges=[0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def euclidean_distance(v1, v2):
    """
    Count euclidean distance of 2 vectors representing points in N-dimensional space.
    :param v1: vector representing point A, iterable, e.g. tuple or list
    :param v2: vector representing point B, iterable, e.g. tuple or list
    """
    distances = [(a - b) ** 2 for a, b in zip(v1, v2)]
    return math.sqrt(sum(distances))


def get_frame_from_video(video_path, frame_time=0.0):
    """From the given video pick 1 frame (image/photo) that is closest to the given time [seconds] since video start."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_tolerance_ms = (1 / fps) / 2 * 1000  # frame duration tolerance [ms]
    frame_time_ms = frame_time * 1000
    current_time_ms = 0.0
    frame = None
    found = False
    while not found:
        ret, frame = cap.read()
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # current time of video [ms]
        if not ret:
            print("Can't load next frame.")
            break

        if current_time_ms-time_tolerance_ms <= frame_time_ms <= current_time_ms+time_tolerance_ms:
            return frame

    if not found:
        raise Exception("Did not find frame with time {} [s] in given video. Video searched for: {} [s]".format(frame_time, current_time_ms/1000))


def save_frame_from_video(src_path, dst_path, frame_time=0.0, rotate_upside_down=False):
    """
    Extract a frame at the given time from the given video and save it as image. Use rotate_upside_down if the frame has been saved upside down. Example:
    save_frame_from_video('../testing_data/s3_m_front_multi.mp4', '../testing_data/s3_m_front_multi_y600.png', frame_time=20.5)
    """
    frame = get_frame_from_video(src_path, frame_time)
    if rotate_upside_down:
        frame = cv2.rotate(frame, 1)
    cv2.imwrite(dst_path, frame)
