import cv2
import PovaPose as pp

""" Usage example"""


def run_single_person_detection():
    frame1 = cv2.imread("test.jpg")
    frame2 = cv2.imread("test2.jpg")

    detector_cam1 = pp.PovaPose()
    detector_cam1.set_image_for_detection(frame1)
    detector_cam1.run_single_person_detection()

    detector_cam1.set_image_for_detection(frame2)
    detector_cam1.run_single_person_detection()


def run_multi_person_detection():
    """
        frame1 = cv2.imread("f1.jpg")
    """

    photos = ["f1.jpg", "f2.jpg"]

    detector_cam1 = pp.PovaPose()

    for idx_photo, p in enumerate(photos):
        print("Result for photo number:" + str(idx_photo))

        frame1 = cv2.imread(p)
        detector_cam1.set_image_for_detection(frame1)
        result = detector_cam1.run_multi_person_detection()

        for i, r, in enumerate(result):
            print("Person numbere:" + str(i))
            print("Nose yx: " + str(r[1]))
            print("Right hip yx:" + str(r[2]))
            print("Left hip yx:" + str(r[3]))

    """ Structure for each person
        [0] - Sub picture for person
        [1] - Nose xy
        [2] - Right hip
        [3] - Left hip
        [4] - Right ankle
        [5] - Left ankle
    """


    """
    cv2.imshow("Detected Pose", r[0])
    cv2.waitKey(0)
    """


def main():
    """run_single_person_detection()"""
    run_multi_person_detection()


if __name__ == '__main__':
    main()
