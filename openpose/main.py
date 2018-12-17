import cv2
import PovaPose as pp
import numpy as np

""" Usage example"""


def get_first_photo_from_video(videao_file_name):
    photos = []

    cap = cv2.VideoCapture(videao_file_name)
    ret, frame = cap.read()

    if not ret:
        print("Video frame can not be loaded.")
        return

    photos.append(frame)
    return photos

def run_multi_person_detection():
    """ OPTION 1: Get Photo from file
        photos = []

        photo_names = ["f1.jpg", "f2.jpg"]
        
        for fileName in photo_names:
            frame1 = cv2.imread(p)
            photos.append(frame1)
    """

    """ OPTION 2: Get Photo from video
        photos = getPhotosFromVideo("v1")
    """

    photos = []
    photo_names = ["data/f1.jpg", "data/f2.jpg"]

    for fileName in photo_names:
        frame = cv2.imread(fileName)
        photos.append(frame)

    detector_cam1 = pp.PovaPose()

    for idx_photo, p in enumerate(photos):
        print("Results for photo number: " + str(idx_photo))

        detector_cam1.set_image_for_detection(p)
        result = detector_cam1.run_multi_person_detection()

        for i, r, in enumerate(result):
            if len(r[1]) == 0 or len(r[2]) == 0 or len(r[3]) == 0:
                print("Person number : " + str(i) + " does not have nose or hip detected.")
                continue

            r_hip = r[2]
            l_hip = r[3]
            average_hip = ((r_hip[0]+l_hip[0])/2, (r_hip[1]+l_hip[1])/2)
            distance = np.linalg.norm(np.asarray(r[1]) - np.asarray(average_hip))

            print("Person number: " + str(i))
            print("Nose y,x: " + str(r[1]))
            print("Average hip y,x: " + str(average_hip))
            print("Nose-hip distance:" + str(distance))

    cv2.waitKey(0)


def get_data_for_calibration(first_image_path, second_image_path):
    frame_f = cv2.imread(first_image_path)
    frame_s = cv2.imread(second_image_path)

    neck_hip_f = []
    neck_hip_s = []

    detector = pp.PovaPose()

    detector.set_image_for_detection(frame_f)
    result = detector.run_multi_person_detection()

    for i, r, in enumerate(result):
        if len(r[1]) == 0 or len(r[2]) == 0 or len(r[3]) == 0:
            print("Person number : " + str(i) + " does not have neck or hip detected.")
            continue

        r_hip = r[2]
        l_hip = r[3]
        average_hip = [(r_hip[0]+l_hip[0])/2, (r_hip[1]+l_hip[1])/2]

        neck_hip_f.append(r[1])
        neck_hip_f.append(average_hip)

    detector.set_image_for_detection(frame_s)
    result = detector.run_multi_person_detection()

    for i, r, in enumerate(result):
        if len(r[1]) == 0 or len(r[2]) == 0 or len(r[3]) == 0:
            print("Person number : " + str(i) + " does not have neck or hip detected.")
            continue

        r_hip = r[2]
        l_hip = r[3]
        average_hip = [(r_hip[0]+l_hip[0])/2, (r_hip[1]+l_hip[1])/2]

        neck_hip_s.append(r[1])
        neck_hip_s.append(average_hip)

    return neck_hip_f, neck_hip_s


def main():
    neck_hip_f, neck_hip_s = get_data_for_calibration("data/f1.jpg", "data/f2.jpg")

    print("First: " + str(neck_hip_f))
    print("Second: " + str(neck_hip_s))


if __name__ == '__main__':
    main()
