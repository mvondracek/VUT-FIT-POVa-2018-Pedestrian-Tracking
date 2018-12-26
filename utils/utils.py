import math

import cv2
import numpy as np


class RectangleSelectionCallback:
    """Serves to select a rectangle in a cv2 image window using mouse. Implementation inspired by VUT FIT POVa du02.py."""
    def __init__(self, image_width, image_height):
        """
        Width and height of the image is used to ignore coordinates larger than the image.
        :param image_width: width of the image used for rectangle selection
        :param image_height: height of the image used for rectangle selection
        """
        self._image_width = image_width
        self._image_height = image_height
        self._point1 = None
        self._point2 = None
        self.top_left = [0, 0]
        self.bottom_right = [0, 0]
        self.selection_active = False
        self.selection_finished = False

    def mouse_callback(self, event, x, y, flags, param):
        # If the left mouse button is clicked, record the starting (x, y) coordinates and indicate that selection started.
        if event == cv2.EVENT_LBUTTONDOWN:
            self._point1 = self._point2 = (int(x), int(y))
            self.selection_active = True
            self._update_corners()
        # If ongoing selection, update rectangle.
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selection_active:  # TODO
                self._point2 = (int(x), int(y))
                self._update_corners()
        # Finish selection when left mouse button is released.
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selection_active:
                self._point2 = (int(x), int(y))
                self.selection_active = False
                self.selection_finished = True
                self._update_corners()

    def _update_corners(self):
        """Calculate and update top-left and bottom-right corner coordinates."""
        self.top_left[0] = max(0, min(self._point1[0], self._point2[0]))
        self.top_left[1] = max(0, min(self._point1[1], self._point2[1]))
        self.bottom_right[0] = min(self._image_width, max(self._point2[0], self._point2[0]))
        self.bottom_right[1] = min(self._image_height, max(self._point2[1], self._point2[1]))


def synchronize_images(image1, image2, mask1=None, mask2=None, interactive=False):  # TODO rename
    """#TODO"""
    # windows for interactive selection
    window1 = None
    window2 = None
    if interactive:
        image1_copy = image1.copy()
        image2_copy = image2.copy()

        window1 = 'image1_interactive_mask_selection'
        window2 = 'image2_interactive_mask_selection'
        cv2.namedWindow(window1, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window2, cv2.WINDOW_NORMAL)

        rectangle1 = RectangleSelectionCallback(image1.shape[1], image1.shape[0])
        rectangle2 = RectangleSelectionCallback(image2.shape[1], image2.shape[0])
        cv2.setMouseCallback(window1, rectangle1.mouse_callback)
        cv2.setMouseCallback(window2, rectangle2.mouse_callback)

        # FIXME after moving to image rpovider uncomment: logger.info("Select a rectangle in both images, then press any key.")
        cv2.imshow(window1, image1_copy)
        cv2.imshow(window2, image2_copy)
        cv2.waitKey()  # wait until the mask area is selected for both images
        assert rectangle1.selection_finished and rectangle2.selection_finished, "Rectangle must be selected in both images!"

        # TODO draw the rectangle

        mask1 = np.zeros(image1.shape[0:2], dtype=np.uint8)  # mask needs to has one dimension less
        mask2 = np.zeros(image2.shape[0:2], dtype=np.uint8)
        mask1[rectangle1.top_left[1]:rectangle1.bottom_right[1]+1, rectangle1.top_left[0]:rectangle1.bottom_right[0]+1] = 1
        mask2[rectangle2.top_left[1]:rectangle2.bottom_right[1]+1, rectangle2.top_left[0]:rectangle2.bottom_right[0]+1] = 1
        # TODO add +1 to all subimage selections

    image1, image2 = synchronize_images_colors(image1, image2, mask1=mask1, mask2=mask2, interactive=False)
    image1, image2 = synchronize_images_light_intensity(image1, image2, mask1=mask1, mask2=mask2, interactive=False)

    if interactive:
        cv2.imshow(window1, image1)
        cv2.imshow(window2, image2)
        cv2.waitKey()

    return image1, image2


def synchronize_images_colors(image1, image2, mask1=None, mask2=None, interactive=False):  # TODO rename
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


def synchronize_images_light_intensity(image1, image2, mask1=None, mask2=None, interactive=False):  # TODO rename
    """TODO"""
    if interactive:
        raise NotImplementedError  # TODO implement manual selection of image mask by mouse
        window1 = 'image1_interactive_mask_selection'
        cv2.namedWindow(window1, cv2.WINDOW_NORMAL)
        window2 = 'image2_interactive_mask_selection'
        cv2.namedWindow(window2, cv2.WINDOW_NORMAL)
        cv2.imshow(window1, image1)
        cv2.imshow(window2, image2)
        cv2.waitKey()

    light_channel = 0
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

    # get mean value of lightness
    mean1, _ = cv2.meanStdDev(image1[:, :, light_channel], mask=mask1)
    mean2, _ = cv2.meanStdDev(image2[:, :, light_channel], mask=mask2)
    ratio = mean1 / mean2
    ratio = ratio.flatten()  # just 1 non-nested value

    image1 = image1.astype(np.float16)
    image2 = image2.astype(np.float16)
    image2[:, :, light_channel] = image2[:, :, light_channel] * ratio
    np.clip(image2, a_min=0, a_max=255, out=image2)

    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_LAB2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_LAB2BGR)

    if interactive:
        cv2.imshow(window1, image1)
        cv2.imshow(window2, image2)
        cv2.waitKey()

    return image1, image2


image1 = cv2.imread('../testing_data/s3_m_front_multi_y600.png')  # TODO Implement image provider.
image2 = cv2.imread('../testing_data/s3_f_side_multi_y600.png')  # TODO Implement image provider.
synchronize_images(image1, image2, None, None, interactive=True)  # TODO


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
