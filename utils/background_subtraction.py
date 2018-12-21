"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""


import numpy as np
import cv2 as cv


# NOTE: https://docs.opencv.org/3.4/db/d5c/tutorial_py_bg_subtraction.html
#
# Problem is that we don't have only background image in the beginning of the video. I tried placing there background
# image from other parts of the video, but exposure settings is different.
#
# Another problem is that M and F cameras change exposure during recording, therefore background image is changing.
#
def mog2():
    fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)
    cap = cv.VideoCapture('../testing_data/s3_m_front_multi.mp4')

    while True:
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        cv.imshow('frame', frame)
        cv.imshow('mask', fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()


# This approach can use single background image and to get areas which are different from the background.
# result is not a whole body, but only parts which differ from the static background.
def subtract():
    bg_color = cv.imread('../testing_data/s3_m_front_multi_bg.png')
    bg = cv.cvtColor(bg_color, cv.COLOR_BGR2GRAY)
    cap = cv.VideoCapture('../testing_data/s3_m_front_multi.mp4')
    while True:
        ret, frame_color = cap.read()
        frame = cv.cvtColor(frame_color, cv.COLOR_BGR2GRAY)
        fgmask = cv.subtract(bg, frame)
        _, fgmask_tresh = cv.threshold(fgmask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        cv.imshow('frame', frame)
        cv.imshow('mask', fgmask_tresh)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    #subtract()
    #mog2()
