""""
Concrete MediaPipe module


# Copyright (c) 2022-Current Alex Estrada <aestradab@ucdavis.edu>
"""

import mediapipe as mp
import cv2


class MediaPipe_Method:

    def __init__(self, references=None):
        self.refs = references
        # landmarks are ordered [left, right] respectively
        self.mouth_index = [61, 291, 76, 306, 62, 292, 78, 308, 191, 415, 80, 310, 95, 324, 88, 318, 184, 408,
                            74, 304, 183, 407, 42, 272, 96, 325, 89, 319, 77, 307, 90, 320, 73, 303, 72, 302, 41,
                            271, 38, 268, 81, 311, 82, 312, 178, 402, 87, 317, 179, 403, 86, 316, 180, 404, 85, 315,
                            57, 287, 185, 409, 40, 270, 39, 269, 37, 267, 146, 375, 91, 321, 181, 405, 84, 314]
        self.eye_index = [225, 445, 224, 444, 223, 443, 222, 442, 221, 441, 33, 263, 246, 466, 161, 388, 160, 387,
                          159, 386, 158, 385, 157, 384, 173, 398, 133, 362, 7, 249, 163, 390, 144, 373, 145, 374,
                          153, 380, 154, 381, 155, 382]
        self.eyebrow_index = [70, 300, 63, 293, 105, 334, 66, 296, 107, 336, 46, 276,
                              53, 283, 52, 282, 65, 295, 55, 285]
        self.all_index = self.mouth_index + self.eye_index + self.eyebrow_index + self.refs
        self.image1 = None
        self.image2 = None

    def image_Input(self, img1, img2):
        dict1 = {}
        dict2 = {}
        index_list = self.all_index

        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils

        with mp_face_mesh.FaceMesh(static_image_mode=True,
                                   max_num_faces=1,
                                   min_detection_confidence=0.5) as face_mesh:
            image1 = img1
            image2 = img2
            height1, width1, _ = image1.shape
            height2, width2, _ = image2.shape

            image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            results1 = face_mesh.process(image1_rgb)
            results2 = face_mesh.process(image2_rgb)

            if results1.multi_face_landmarks is not None:
                for face_landmarks in results1.multi_face_landmarks:
                    for i in index_list:
                        x = int(face_landmarks.landmark[i].x * width1)
                        y = int(face_landmarks.landmark[i].y * height1)
                        # print("IMG1", i, face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                        # print(i,":", face_landmarks.landmark[i].z)
                        dict1[i] = [x, y]
                        cv2.circle(image1, (x, y), 2, (255, 0, 0), 2)  # BRG

            if results2.multi_face_landmarks is not None:
                for face_landmarks in results2.multi_face_landmarks:
                    for i in index_list:
                        x = int(face_landmarks.landmark[i].x * width2)
                        y = int(face_landmarks.landmark[i].y * height2)
                        # print("IMG2", i, face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                        # print(i, ":", face_landmarks.landmark[i].z)
                        dict2[i] = [x, y]
                        cv2.circle(image2, (x, y), 2, (255, 0, 0), 2)  # BRG

            self.image1 = image1
            self.image2 = image2

        # cv2.imshow("Image", image1)
        # cv2.imshow("image2", image2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("FaceMesh Landmarks Collected")
        return [dict1, dict2]
