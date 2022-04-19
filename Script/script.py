"""
NeuroVA Script


# Copyright (c) 2022-Current Alex Estrada <aestradab@ucdavis.edu>
"""

from Code.face_mesh_mediapipe import MediaPipe_Method
from GCM.geometric_computation import Geometric_Computation
import numpy as np
import cv2
import matplotlib.pyplot as plt


# loading files
# img1 = cv2.imread('../Images/neutral_face_girl.jpg')
# # img11 = cv2.flip(img1, 1)   # flipped image
# img2 = cv2.imread('../Images/surprised_girl_face.jpg')
# # img22 = cv2.flip(img2, 1)   # flipped image 2
#
# img3 = cv2.imread('../Images/SahandNeutral.jpg')
# img35 = cv2.imread('../Images/SahandNeutralcropped1.jpg')
# img4 = cv2.imread('../Images/SahandWow.jpg')
#
img5 = cv2.resize(cv2.imread('../Images/trialNeutral.jpg'), (3088, 3088))
img6 = cv2.resize(cv2.imread('../Images/trialWow.jpg'), (3088, 3088))
#
# img7 = cv2.imread('../Images/neutralJavier.jpg')
# img8 = cv2.imread('../Images/wowJavier.jpg')
# # original Javier = 3088 by 2316
#
# img9 = cv2.imread('../Images/georgeclooneyNeutral.jpg')
# img10 = cv2.imread('../Images/georgeclooneySmiling.jpg')
#
# img11 = cv2.imread('../Images/bradpittNeutral.jpg')
# img12 = cv2.imread('../Images/bradpittSmiling.jpg')
#
# img13 = cv2.imread('../Images/LaurenNeutral.jpg')
# img14 = cv2.imread('../Images/LaurenSmile.jpg')
# img15 = cv2.imread('../Images/LaurenWow.jpg')
# img16 = cv2.imread('../Images/LaurenFrown.jpg')
# # # print(img13.shape)
# #
# img17 = cv2.imread('../Images/GigiNeutral.jpg')
# img18 = cv2.imread('../Images/GigiSmile.jpg')
# img19 = cv2.imread('../Images/GigiWow.jpg')
# img20 = cv2.imread('../Images/GigiFrown.jpg')
# # # # print(img17.shape)
# #
img21 = cv2.imread('../Images/KelseyNeutral.jpg')
img22 = cv2.imread('../Images/KelseySmile.jpg')
img23 = cv2.imread('../Images/KelseyWow.jpg')
img24 = cv2.imread('../Images/KelseyFrown.jpg')
# # print(img21.shape)
#
# img25 = cv2.imread('../Images/AlexNeutral.jpg')
# img26 = cv2.imread('../Images/AlexSmile.jpg')
# img27 = cv2.imread('../Images/AlexWow.jpg')
# img28 = cv2.imread('../Images/AlexFrown.jpg')
# # print(img25.shape)
#
# img29 = cv2.resize(cv2.imread('../Images/trial2Neutral.jpg'), (3088, 3088))
#
# img30 = cv2.imread('../Images/GenNeutralSquare.jpg')
# print(img30.shape)

# img31 = cv2.imread('../Images/KoushNeutral.jpg')
# img32 = cv2.imread('../Images/KoushSmiling.jpg')
# img33 = cv2.imread('../Images/KoushWow.jpg')
# img34 = cv2.imread('../Images/KoushFrown.jpg')
#
# img35 = cv2.imread('../Images/NicoleNeutral.jpg')
# img36 = cv2.imread('../Images/NicoleSmile.jpg')
# img37 = cv2.imread('../Images/NicoleWow.jpg')
# img38 = cv2.imread('../Images/NicoleFrown.jpg')
#
# img39 = cv2.imread('../Images/JordanNeutral.jpg')
# img40 = cv2.imread('../Images/JordanSmile.jpg')
# img41 = cv2.imread('../Images/JordanWow.jpg')
# img42 = cv2.imread('../Images/JordanFrown.jpg')

# img43 = cv2.imread('../Images/LoveNeutral.jpg')
# img44 = cv2.imread('../Images/LoveSmile.jpg')
# img45 = cv2.imread('../Images/LoveWow.jpg')
# img46 = cv2.imread('../Images/LoveFrown.jpg')
#
# img47 = cv2.imread('../Images/JavierPNeutral.jpg')
# img48 = cv2.imread('../Images/JavierPSmile.jpg')
# img49 = cv2.imread('../Images/JavierPWow.jpg')
# img50 = cv2.imread('../Images/JavierPFrown.jpg')
#
# img51 = cv2.imread('../Images/ValeriaNeutral.jpg')
# img52 = cv2.imread('../Images/ValeriaSmile.jpg')
# img53 = cv2.imread('../Images/ValeriaWow.jpg')
# img54 = cv2.imread('../Images/ValeriaFrown.jpg')

# refs
refs_top_bot = [10, 152]
refs_sides = [127, 356]

# call MediaPipe, initialize indexes, reference points, and all required images
# image_test = MediaPipe_Method(refs_sides, [img3, img13, img17, img21, img25, img5])
# image_test = MediaPipe_Method(refs_sides, [img6, img21, img25, img29, img30])
# image_test = MediaPipe_Method(refs_sides, [img3, img35, img4              # Sahand (img35 is cropped)
image_test = MediaPipe_Method(refs_sides, [img5, img6])                       # trial
# image_test = MediaPipe_Method(refs_sides, [img13, img14, img15, img16])       # Lauren
# image_test = MediaPipe_Method(refs_sides, [img21, img22, img23, img24])     # Kelsey
# image_test = MediaPipe_Method(refs_sides, [img17, img18, img19, img20])     # Gigi
# image_test = MediaPipe_Method(refs_sides, [img25, img26, img27, img28])     # Alex
# image_test = MediaPipe_Method(refs_sides, [img31, img32, img33, img34])       # Koush
# image_test = MediaPipe_Method(refs_sides, [img35, img36, img37, img38])     # Nicole
# image_test = MediaPipe_Method(refs_sides, [img39, img40, img41, img42])         # Jordan
# image_test = MediaPipe_Method(refs_sides, [img43, img44, img45, img46])     # Genevieve
# image_test = MediaPipe_Method(refs_sides, [img47, img48, img49, img50])     # Javier
# image_test = MediaPipe_Method(refs_sides, [img51, img52, img53, img54])     # Valeria
original_dicts, mirrored_dicts, mp_imgs, mp_mirrored_imgs = image_test.mp_run("", 0)    # name, to save CSV=True
# original dicts: ----- dictionaries of the landmarks of original photos
# mirrored dicts: ----- dictionaries of the landmarks of mirrored images
# mp_imgs: ------------ edited image of the original photo with landmarks
# mp_mirrored_imgs: --- edited image of the mirrored photo with landmarks

# scale_percent = 40  # percent of original size
# mp_resized = []
# for i in mp_imgs:
#     width = int(i.shape[1] * scale_percent / 100)
#     height = int(i.shape[0] * scale_percent / 100)
#     dim = (width, height)
#
#     # resize image
#     resized = cv2.resize(i, dim, interpolation=cv2.INTER_AREA)
#     mp_resized.append(resized)
#
# cv2.imshow("R0", mp_resized[0])
# cv2.imshow("R1", mp_resized[1])
# cv2.imshow("R2", mp_resized[2])
# cv2.imshow("R3", mp_resized[3])
# cv2.imshow("R4", mp_resized[4])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# initiating GCM with original and mirrored dictionaries. References and sagittal points popped upon initialization.
patient = Geometric_Computation(original_dicts)
mirrored_patient = Geometric_Computation(mirrored_dicts)

# alternate model
original_mirrored_dist, all_avg, upper_lower_split_avg = patient.get_icp_distances(mirrored_patient.norm_array_dicts)

for idx, i in enumerate(original_mirrored_dist):
    print("----------------------------------------ALTERNATE MODEL RESULTS-----------------------------------------")
    print("IMG", idx+1, ":")
    # print("All original-to-mirrored distances:", i)
    print("Upper Asymmetry Score:", upper_lower_split_avg[idx][0]*10000)
    print("Lower Asymmetry Score:", upper_lower_split_avg[idx][1]*10000)
    print("Weighted Average:", all_avg[idx]*10000)

# patient.GCM1()
patient.GCM2()
