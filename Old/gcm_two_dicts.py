"""
Concrete MediaPipe module


# Copyright (c) 2022-Current Alex Estrada <aestradab@ucdavis.edu>
"""

import math
import numpy as np


class Geometric_Computation:

    def __init__(self, dicts=None, ref=None):
        self.dicts = dicts
        self.refs = ref
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
        self.all_index = self.mouth_index + self.eye_index + self.eyebrow_index
        # references
        self.ref_im1 = dicts[0].pop(ref[0]), dicts[0].pop(ref[1])
        self.ref_im2 = dicts[1].pop(ref[0]), dicts[1].pop(ref[1])
        self.norm_array_dict1 = None
        self.norm_array_dict2 = None
        self.results = None

    def show_dicts(self):
        dict1, dict2 = self.dicts[0], self.dicts[1]
        return print("Image 1 Values:", dict1, "\nImage 2 Values:", dict2)

    def show_refs(self):
        return print("Image 1 Ref Values:", self.ref_im1, "\nImage 2 Ref Values:", self.ref_im2)

    def factor_dicts(self):
        # getting refs distance
        difs1 = np.subtract(np.array(self.ref_im1[0]), np.array(self.ref_im1[1]))
        difs2 = np.subtract(np.array(self.ref_im2[0]), np.array(self.ref_im2[1]))
        factor1 = math.sqrt(difs1[0] ** 2 + difs1[1] ** 2)
        factor2 = math.sqrt(difs2[0] ** 2 + difs2[1] ** 2)
        # normalizing with respect to refs
        numpy_array_dict1 = np.array(list(self.dicts[0].values()))
        numpy_array_dict2 = np.array(list(self.dicts[1].values()))
        self.norm_array_dict1 = numpy_array_dict1 / factor1
        self.norm_array_dict2 = numpy_array_dict2 / factor2
        print("Dictionaries are now normalized with respect to:")
        print("1:", self.ref_im1, "Factor:", factor1, "\n2:", self.ref_im2, "Factor:", factor2)

    def all_diffs(self):
        prim = np.subtract(self.norm_array_dict2, self.norm_array_dict1)**2
        sec = []
        for i in range(0, len(prim)):
            sec.append(math.sqrt(prim[i][0] + prim[i][1]))
        dr = np.array(sec[::2])
        dl = np.array(sec[1::2])
        r = abs(1 - dl / dr)     # len(r) = 68
        u_r = np.sum(r) / len(prim)
        self.results = [dr, dl, r, u_r]
        return print("Geometric Difference Complete")

    def regional_results(self):
        pass

    def show_results(self):
        print("Geometric Computation for landmarks:", self.all_index,
              "\nReference Landmarks:", self.refs,
              "\nAvg. Left Alteration:", np.mean(self.results[1]),
              "\nRight Alteration:", np.mean(self.results[0]),
              "\nR-Value:", np.mean(self.results[2]),
              "\nu_r Value:", self.results[3])
