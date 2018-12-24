import math

import cv2
import numpy as np


def synchronize_colors(image1, image2, point1, point2):
    # TODO zkusit proovnavat body, rectangle prumer, nebo prevest do jineho formatu a upravit jas - HSV / LAB
    #image1 = cv2.imread('../testing_data/s3_m_front_multi_y600.png')  # TODO Implement image provider.
    #image2 = cv2.imread('../testing_data/s3_f_side_multi_y600.png')  # TODO Implement image provider.

    window1 = 'img111'
    cv2.namedWindow(window1, cv2.WINDOW_NORMAL)
    window2 = 'img222'
    cv2.namedWindow(window2, cv2.WINDOW_NORMAL)
    cv2.imshow(window1, image1)
    cv2.imshow(window2, image2)
    #cv2.waitKey()
    #################################################

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

    cv2.imshow(window1, image1)
    cv2.imshow(window2, image2)
    #cv2.waitKey()

    return image1, image2


#synchronize_colors(None, None, None, None)  # TODO

def synchronize_colors_TODOjedenpixel_barvy(image1, image2, point1, point2):
    """
    TODO
    :param image1:
    :param image2:
    :param point1:
    :param point2:
    :return:
    """
    image1 = cv2.imread('../testing_data/s3_m_front_multi_y600.png')  # TODO Implement image provider.
    image2 = cv2.imread('../testing_data/s3_f_side_multi_y600.png')  # TODO Implement image provider.
    # TODO zkusit proovnavat body, rectangle prumer, nebo prevest do jineho formatu a upravit jas
    point1 = (1190, 590)
    point2 = (135, 480)

    window1 = 'img111'
    cv2.namedWindow(window1, cv2.WINDOW_NORMAL)
    window2 = 'img222'
    cv2.namedWindow(window2, cv2.WINDOW_NORMAL)
    cv2.imshow(window1, image1)
    cv2.imshow(window2, image2)
    cv2.waitKey()

    image1 = image1.astype(np.float16)
    image2 = image2.astype(np.float16)
    pixel1 = image1[point1[1], point1[0]]
    pixel2 = image2[point1[1], point1[0]]
    print(image1.dtype)
    print(pixel1)
    print(pixel2)
    ratios = pixel1 / pixel2
    print(ratios)

    image2 = image2[:, :] * ratios

    #frame = np.clip(image2, 0, 255)
    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)

    pixel2 = image2[point1[1], point1[0]]
    print(pixel2)

    cv2.imshow(window1, image1)
    cv2.imshow(window2, image2)
    cv2.waitKey()


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
