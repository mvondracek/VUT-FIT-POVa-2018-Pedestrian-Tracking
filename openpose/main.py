import cv2
import PovaPose as pp

""" Usage example"""


def run_single_person_detection():
    frame1 = cv2.imread("ja11.jpg")
    frame2 = cv2.imread("ja22.jpg")

    detector_cam1 = pp.PovaPose()
    detector_cam1.set_image_for_detection(frame1)
    detector_cam1.run_single_person_detection()

    detector_cam1.set_image_for_detection(frame2)
    detector_cam1.run_single_person_detection()


def run_multi_person_detection():
    frame1 = cv2.imread("group.jpg")

    detector_cam1 = pp.PovaPose()
    detector_cam1.set_image_for_detection(frame1)
    detector_cam1.run_multiple_person_detection()


def main():
    """run_single_person_detection()"""
    run_multi_person_detection()


if __name__ == '__main__':
    main()
