import cv2
import openpose.PovaPose as pp
import numpy as np
import random

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


def run_multi_person_detection_example():
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

        detector_cam1.show()

        for i, r, in enumerate(result):
            distance = np.linalg.norm(np.asarray(r[1]) - np.asarray(r[2]))

            print("Person number: " + str(i))
            print("Nose y,x: " + str(r[1]))
            print("Average hip y,x: " + str(r[2]))
            print("Neck-hip distance:" + str(distance))


def get_data_for_calibration(first_image_path, second_image_path):
    frame_f = cv2.imread(first_image_path)
    frame_s = cv2.imread(second_image_path)

    neck_hip_f = []
    neck_hip_s = []

    detector = pp.PovaPose()

    detector.set_image_for_detection(frame_f)
    result = detector.run_multi_person_detection()

    for i, r, in enumerate(result):
        neck_hip_f.append(r[1])
        neck_hip_f.append(r[2])

    detector.set_image_for_detection(frame_s)
    result = detector.run_multi_person_detection()

    for i, r, in enumerate(result):
        neck_hip_s.append(r[1])
        neck_hip_s.append(r[2])

    return neck_hip_f, neck_hip_s


def calibration_example():
    neck_hip_f, neck_hip_s = get_data_for_calibration("data/f1.jpg", "data/f2.jpg")

    print("First: " + str(neck_hip_f))
    print("Second: " + str(neck_hip_s))


def person_synchronization(first_camera_imgs, second_camera_imgs):
    f_histograms = []
    s_histograms = []
    result = {}

    for f_c_img in first_camera_imgs:
        f_image = cv2.cvtColor(f_c_img, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([f_image], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        dst = np.zeros(hist.shape)
        hist = cv2.normalize(hist, dst)
        flatten = hist.flatten()
        f_histograms.append(flatten)

    for s_c_img in second_camera_imgs:
        s_image = cv2.cvtColor(s_c_img, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([s_image], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        dst = np.zeros(hist.shape)
        hist = cv2.normalize(hist, dst)
        flatten = hist.flatten()
        s_histograms.append(flatten)

    for idx_f, f_hist in enumerate(f_histograms):
        max_match = 0
        idx_of_match = 0
        for idx_s, s_hist in enumerate(s_histograms):
            match = cv2.compareHist(f_hist, s_hist, cv2.HISTCMP_INTERSECT)
            if max_match < match:
                max_match = match
                idx_of_match = idx_s

        result[idx_f] = idx_of_match

    return result

def image_synchronization_example():
    frame1 = cv2.imread("data/group.jpg")
    frame2 = cv2.imread("data/group.jpg")

    detector_cam1 = pp.PovaPose()

    cam1_images = []
    cam2_images = []

    detector_cam1.set_image_for_detection(frame1)
    result = detector_cam1.run_multi_person_detection()
    for i, person in enumerate(result):
        if len(person[0]) == 0 or len(person[1]) == 0 or len(person[2]) == 0 or len(person[3]) == 0:
            print("Person number : " + str(i) + " does not have neck or hip detected.")
            continue
        cam1_images.append(person[0])

    detector_cam1.set_image_for_detection(frame2)
    result = detector_cam1.run_multi_person_detection()
    for i, person in enumerate(result):
        if len(person[0]) == 0 or len(person[1]) == 0 or len(person[2]) == 0 or len(person[3]) == 0:
            print("Person number : " + str(i) + " does not have neck or hip detected.")
            continue
        cam2_images.append(person[0])

    random.shuffle(cam2_images)
    synchronization = person_synchronization(cam1_images, cam2_images)


def main():
    """
        run_multi_person_detection_example()
        calibration_example()
        image_synchronization_example()
    """


if __name__ == '__main__':
    main()
