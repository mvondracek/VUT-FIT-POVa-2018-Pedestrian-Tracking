"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import math

import cv2
import numpy as np


def select_rectangle_mask_using_mouse(window_name, image):
    """
    Display the given image in a window with the give name. Use mouse to draw a rectangular mask in the image. Press 'ESC' to quit (no mask created).
    :return: None if quit by 'ESC', otherwise rectangle mask for the given image
    """
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:  # any window property returns -1.0 if window doesn't exist
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    rect_selection = RectangleSelectionCallback(image.shape[1], image.shape[0])
    cv2.setMouseCallback(window_name, rect_selection.mouse_callback)
    while not rect_selection.selection_finished:
        image_copy = image.copy()
        cv2.rectangle(image_copy, tuple(rect_selection.top_left), tuple(rect_selection.bottom_right), color=(150, 0, 150), thickness=3)
        cv2.imshow(window_name, image_copy)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESCAPE to cancel selection
            break

    if not rect_selection.selection_finished:
        return None

    mask = np.zeros(image.shape[0:2], dtype=np.uint8)  # mask needs to has one dimension less
    mask[rect_selection.top_left[1]:rect_selection.bottom_right[1] + 1, rect_selection.top_left[0]:rect_selection.bottom_right[0] + 1] = 1
    return mask


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
        """This callback should be bind to the target window, e.g. cv2.setMouseCallback("Window1", rect_selection.mouse_callback)"""
        # If the left mouse button is clicked, record the starting (x, y) coordinates and indicate that selection started.
        if event == cv2.EVENT_LBUTTONDOWN:
            self._point1 = self._point2 = (int(x), int(y))
            self.selection_active = True
            self._update_corners()
        # If ongoing selection, update rectangle.
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selection_active:
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
        self.bottom_right[0] = min(self._image_width, max(self._point1[0], self._point2[0]))
        self.bottom_right[1] = min(self._image_height, max(self._point1[1], self._point2[1]))


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
    found = False
    while not found:
        ret, frame = cap.read()
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # current time of video [ms]
        if not ret:
            print("Can't load next frame.")
            break

        if current_time_ms-time_tolerance_ms <= frame_time_ms <= current_time_ms+time_tolerance_ms:
            cap.release()
            return frame

    cap.release()
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
