"""
Concrete MediaPipe module
"""

# Copyright (c) 2022-Current Alex Estrada <aestradab@ucdavis.edu>

from GCM.Geometric_computation import Geometric_Computation
import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# points of interest:
# 6 = reference point
# 66 and 296 = symmetric landmarks
index_list = [6, 66, 296]
dict1 = {}
dict2 = {}

with mp_face_mesh.FaceMesh(static_image_mode=True,
                           max_num_faces=1,
                           min_detection_confidence=0.5) \
                                            as face_mesh:
    # image1 = cv2.imread("SahandNeutral.jpg")
    # image2 = cv2.imread("SahandWow.jpg")

    image1 = cv2.imread("neutralface.jpg")
    image2 = cv2.imread("surprisedFace.jpg")
    print(image1)

    # image1 = cv2.imread("trialNeutral.jpg")
    # image2 = cv2.imread("trialWow.jpg")

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
                cv2.circle(image1, (x,y), 2, (255, 0, 0), 2) # BRG

    if results2.multi_face_landmarks is not None:
        for face_landmarks in results2.multi_face_landmarks:
            for i in index_list:
                x = int(face_landmarks.landmark[i].x * width2)
                y = int(face_landmarks.landmark[i].y * height2)
                # print("IMG2", i, face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                # print(i, ":", face_landmarks.landmark[i].z)
                dict2[i] = [x, y]
                cv2.circle(image2, (x,y), 2, (255, 0, 0), 2) # BRG

    # patient = Geometric_Computation([dict1, dict2], 6, index_list)
    # patient.show_dicts()
    # patient.compute_diffs()

    # cv2.imshow("Image", image1)
    # cv2.imshow("image2", image2)
    # cv2.waitKey(0)
cv2.destroyAllWindows()
