"""
NeuroVA Script


# Copyright (c) 2022-Current Alex Estrada <aestradab@ucdavis.edu>
"""

from Code.face_mesh_mediapipe import MediaPipe_Method
from GCM.geometric_computation import Geometric_Computation
import cv2
import matplotlib.pyplot as plt


# loading files
img1 = cv2.imread('../Images/neutral_face_girl.jpg')
img11 = cv2.flip(img1, 1)   # flipped image
img2 = cv2.imread('../Images/surprised_girl_face.jpg')
img22 = cv2.flip(img2, 1)   # flipped image 2

img3 = cv2.imread('../Images/SahandNeutral.jpg')
img35 = cv2.imread('../Images/SahandNeutralcropped1.jpg')
img4 = cv2.imread('../Images/SahandWow.jpg')

img5 = cv2.resize(cv2.imread('../Images/trialNeutral.jpg'), (500, 500))
img6 = cv2.resize(cv2.imread('../Images/trialWow.jpg'), (500, 500))

img7 = cv2.imread('../Images/neutralJavier.jpg')
img8 = cv2.imread('../Images/wowJavier.jpg')
# original Javier = 3088 by 2316

# refs
refs_top_bot = [10, 152]
refs_sides = [127, 356]

# call MediaPipe, initialize indexes and 2 reference points
# return landmark dictionaries using image input
image_test = MediaPipe_Method(refs_sides, [img1, img2, img3, img4])
original_dicts, mirrored_dicts, mp_imgs, mp_mirrored_imgs = image_test.mp_run()

# cv2.imshow("IMG", img_out)
# cv2.imshow("IMG2", img1_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# GCM only computed fully with 2 images
patient = Geometric_Computation(original_dicts)
mirrored_patient = Geometric_Computation(mirrored_dicts)

patient.mid_norm_plot(0)    # Default is true to plot
mirrored_patient.mid_norm_plot(0)

patient.get_icp_distances(mirrored_patient.norm_array_dicts)

# patient.left_right_diffs()
# patient.get_icp_distances()   # only takes two images from the self.dictionary right now

# patient.all_diffs()
# patient.show_results()
