#!/usr/bin/env python3
"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology

Basic distance to camera of an object with known dimensions computed based on a reference image.

Notes:
    https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
    https://photo.stackexchange.com/questions/12434/how-do-i-calculate-the-distance-of-an-object-in-a-photo
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
"""
import cv2


class Measurement:
    def __init__(self):
        self.start_point = None
        self.stop_point = None
        self.measuring = False
        self.finished = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = self.stop_point = (int(x), int(y))
            self.measuring = True
            self.finished = False
        elif event == cv2.EVENT_MOUSEMOVE and self.measuring:
            self.stop_point = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONUP and self.measuring:
            self.stop_point = (int(x), int(y))
            self.measuring = False
            self.finished = True

    def reset(self):
        self.start_point = None
        self.stop_point = None
        self.measuring = False
        self.finished = False

    @property
    def width(self):
        return abs(self.stop_point[0] - self.start_point[0])


def distance_to_camera(known_width, focal_length, per_width):
    """
    https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
    """
    # compute and return the distance from the maker to the camera
    return (known_width * focal_length) / per_width


def main():
    # region known object configuration, change according to your object
    known_distance = 150  # distance in mm
    known_width = 77  # horizontal width in mm
    # endregion

    cap = cv2.VideoCapture(0)

    cv2.namedWindow('distance_to_camera')
    measurement = Measurement()
    cv2.setMouseCallback('distance_to_camera', measurement.mouse_callback)

    print('Place {0} mm object {1} mm from the camera and measure horizontally.'.format(known_width, known_distance))

    while True:
        ret, image_drawing = cap.read()
        if measurement.measuring:
            cv2.line(image_drawing, measurement.start_point, measurement.stop_point, (0, 255, 0))
            print('measured_width={0} px'.format(measurement.width))
        if measurement.finished:
            break
        cv2.imshow('distance_to_camera', image_drawing)
        cv2.waitKey(20)
    focal_length = (measurement.width * known_distance) / known_width
    print('focal_length={0}'.format(focal_length))
    measurement.reset()

    print('Place {0} mm object somewhere from the camera and measure horizontally.'.format(known_width))
    while True:
        ret, image_drawing = cap.read()
        if measurement.measuring:
            cv2.line(image_drawing, measurement.start_point, measurement.stop_point, (0, 255, 0))
            if measurement.width != 0:
                distance = distance_to_camera(known_width, focal_length, measurement.width)
                print('distance={0} mm'.format(distance))
        if measurement.finished:
            break
        cv2.imshow('distance_to_camera', image_drawing)
        cv2.waitKey(20)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
