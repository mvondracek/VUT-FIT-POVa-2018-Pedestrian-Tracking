import cv2
import numpy as np


class PovaPose:
    def __init__(self):
        self.net = self._net_initialization()
        self.threshold = 0.1
        self.frameCopy = None
        self.frameWidth = 0
        self.frameHeight = 0

        """ Single person """
        """ Nose, Right ankle, Left ankle """
        self.main_points = [0, 10, 13]
        self.main_points_multi = [12, 6, 9, 8, 11]

        self.POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                           [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

        """ Group of people """
        self.detected_keypoints = []
        self.nPoints = 18
        self.keypoints_list = np.zeros((0, 3))

        self.MULTI_POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                                 [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                                 [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
                                 [2, 17], [5, 16]]

        self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
                       [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
                       [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
                       [37, 38], [45, 46]]

        self.colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
                  [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
                  [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

    @staticmethod
    def _net_initialization():
        proto_file = "pose/coco/pose_deploy_linevec.prototxt"
        weights_file = "pose/coco/pose_iter_440000.caffemodel"
        return cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    def set_image_for_detection(self, cv2_image):
        frame = cv2_image
        self.frameCopy = np.copy(frame)
        self.frameWidth = frame.shape[1]
        self.frameHeight = frame.shape[0]

        in_height = 368
        in_width = int((in_height / self.frameHeight) * self.frameWidth)
        inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)
        # Fix the input Height and get the width according to the Aspect Ratio

        self.net.setInput(inp_blob)

    def run_single_person_detection(self):
        """
        Run detection on single person.
        :return: Array of three point that represent 1.nose, 2.right ankle, 3.left ankle
        """

        """ The 'output' is a 4D matrix :
        The first dimension being the image ID ( in case you pass more than one image to the network ).
        The second dimension indicates the index of a keypoint. The model produces Confidence Maps and Part Affinity maps which are all concatenated.
            For COCO model it consists of 57 parts
                – 18 keypoint confidence Maps + 1 background
                – 19*2 Part Affinity Maps. Similarly
        The third dimension is the height of the output map.
        The fourth dimension is the width of the output map.
        """
        output = self.net.forward()

        h = output.shape[2]
        w = output.shape[3]

        # Empty list to store the detected keypoints
        points = []

        for i in self.main_points:
            # confidence map of corresponding body's part.
            prob_map = output[0, i, :, :]

            # Find global maxima of the prob_map.
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            # Scale the point to fit on the original image
            x = (self.frameWidth * point[0]) / w
            y = (self.frameHeight * point[1]) / h

            if prob > self.threshold:
                cv2.circle(self.frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(self.frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        cv2.imshow("Output-Keypoints", self.frameCopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return points

    def run_multi_person_detection(self):
        """ Structure for each person
            [0] - Sub picture for person
            [1] - Nose xy
            [2] - Right hip
            [3] - Left hip
            [4] - Right ankle
            [5] - Left ankle
        """
        output = self.net.forward()

        keypoint_id = 0
        frame_copy_2 = self.frameCopy

        for part in range(self.nPoints):
            probMap = output[0, part, :, :]
            probMap = cv2.resize(probMap, (frame_copy_2.shape[1], frame_copy_2.shape[0]))
            keypoints = self.getKeypoints(probMap)
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                self.keypoints_list = np.vstack([self.keypoints_list, keypoints[i]])
                keypoint_id += 1

            self.detected_keypoints.append(keypoints_with_id)

        frameClone = frame_copy_2.copy()
        valid_pairs, invalid_pairs = self.getValidPairs(output)
        personwiseKeypoints = self.getPersonwiseKeypoints(valid_pairs, invalid_pairs)

        return self.getResultForEachPerson(personwiseKeypoints, frameClone)

    def getResultForEachPerson(self, personwiseKeypoints, frameClone):
        people = []

        for n in range(len(personwiseKeypoints)):
            leftTopPoint = [self.frameWidth, self.frameHeight]
            rightBottomPoint = [0, 0]
            for i in range(17):
                index = personwiseKeypoints[n][np.array(self.MULTI_POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(self.keypoints_list[index.astype(int), 0])
                A = np.int32(self.keypoints_list[index.astype(int), 1])

                if leftTopPoint[0] > B[0]:
                    leftTopPoint[0] = B[0]
                if leftTopPoint[0] > B[1]:
                    leftTopPoint[0] = B[1]

                if leftTopPoint[1] > A[0]:
                    leftTopPoint[1] = A[0]
                if leftTopPoint[1] > A[1]:
                    leftTopPoint[1] = A[1]

                if rightBottomPoint[0] < B[1]:
                    rightBottomPoint[0] = B[1]
                if rightBottomPoint[0] < B[0]:
                    rightBottomPoint[0] = B[0]

                if rightBottomPoint[1] < A[0]:
                    rightBottomPoint[1] = A[0]
                if rightBottomPoint[1] < A[1]:
                    rightBottomPoint[1] = A[1]

                """
                    cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), self.colors[i], 3, cv2.LINE_AA)
                """
            person = self.get_sub_image(leftTopPoint, rightBottomPoint, frameClone)
            structure = [[] for count in range(6)]
            """ Structure for each person
                [0] - Sub picture for person
                [1] - Nose xy
                [2] - Right hip
                [3] - Left hip
                [4] - Right ankle
                [5] - Left ankle
            """
            structure[0] = person
            for idx, p in enumerate(self.main_points_multi):
                index = personwiseKeypoints[n][np.array(self.MULTI_POSE_PAIRS[p])]
                if -1 in index:
                    continue
                B = np.int32(self.keypoints_list[index.astype(int), 0])
                A = np.int32(self.keypoints_list[index.astype(int), 1])

                """Save point"""
                structure[idx + 1] = [B[1], A[1]]

                cv2.circle(self.frameCopy, (int(B[1]), int(A[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

            cv2.rectangle(self.frameCopy, (leftTopPoint[0], leftTopPoint[1]), (rightBottomPoint[0], rightBottomPoint[1]), (255,0,0))
            people.append(structure)

        people = people
        cv2.imshow("Detected Pose", self.frameCopy)
        cv2.waitKey(0)
        return people

    def get_sub_image(self, left_top_point, right_bottom_point, frame_clone):
        return frame_clone[left_top_point[1]:right_bottom_point[1], left_top_point[0]:right_bottom_point[0]]

    def getKeypoints(self, probMap):
        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

        mapMask = np.uint8(mapSmooth > self.threshold)
        keypoints = []

        # find the blobs
        _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

        # x a y for each blob containing the body part
        return keypoints

    def getValidPairs(self, output):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7
        # loop for every POSE_PAIR
        for k in range(len(self.mapIdx)):
            # A->B constitute a limb
            pafA = output[0, self.mapIdx[k][0], :, :]
            pafB = output[0, self.mapIdx[k][1], :, :]
            pafA = cv2.resize(pafA, (self.frameWidth, self.frameHeight))
            pafB = cv2.resize(pafB, (self.frameWidth, self.frameHeight))

            # Find the keypoints for the first and second limb
            candA = self.detected_keypoints[self.MULTI_POSE_PAIRS[k][0]]
            candB = self.detected_keypoints[self.MULTI_POSE_PAIRS[k][1]]
            nA = len(candA)
            nB = len(candB)

            # If keypoints for the joint-pair is detected
            # check every joint in candA with every joint in candB
            # Calculate the distance vector between the two joints
            # Find the PAF values at a set of interpolated points between the joints
            # Use the above formula to compute a score to mark the connection valid

            if (nA != 0 and nB != 0):
                valid_pair = np.zeros((0, 3))
                for i in range(nA):
                    max_j = -1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                               pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores) / len(paf_scores)

                        # Check if the connection is valid
                        # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                        if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else:  # If no keypoints are detected
                print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])
        return valid_pairs, invalid_pairs

    # This function creates a list of keypoints belonging to each person
    # For each detected valid pair, it assigns the joint(s) to a person
    def getPersonwiseKeypoints(self, valid_pairs, invalid_pairs):
        # the last number in each row is the overall score
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(self.mapIdx)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:,0]
                partBs = valid_pairs[k][:,1]
                indexA, indexB = np.array(self.MULTI_POSE_PAIRS[k])

                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += self.keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
                        row[-1] = sum(self.keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints