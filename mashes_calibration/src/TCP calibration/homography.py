#!/usr/bin/env python

import cv2
import numpy as np
from utils import Utils


class Homography():
    def __init__(self, pnts_pattern=np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])):
        self.process = Utils()
        self.pnts_pattern = pnts_pattern

    def read_image(self, image, scale=1):
        size_y, size_x = image.shape[:2]
        image_resized = cv2.resize(image, (size_x*scale, size_y*scale), interpolation=cv2.INTER_NEAREST)
        # Show image and wait for 2 clicks.
        cv2.imshow("Image", image_resized)
        pts_image_resized = self.process.points_average(image_resized)
        pts_image = pts_image_resized/scale
        return pts_image

    def calculate(self, pnts):
        hom, status_1 = cv2.findHomography(pnts, self.pnts_pattern)
        return hom

    def scale(self, hom):
        s = np.mean([hom[0, 0], hom[1, 1]])
        return s

    def transform(self, hom, pnts):
        pnts = np.float32([
            np.dot(hom, np.array([pnt[0], pnt[1], 1])) for pnt in pnts])
        pnts = np.float32([pnt / pnt[2] for pnt in pnts])
        return pnts[:, :2]



if __name__ == '__main__':
    pnts_pattern = np.float32([[0, 0], [6, 0], [0, 6], [6, 6]])
    pnts = np.float32([[8, 7], [25, 6], [9, 24], [25, 23]])
    h = Homography(pnts_pattern)
    hom = h.calculate(pnts)
    print pnts
    print hom, h.scale(hom)
    print np.around(h.transform(hom, pnts), decimals=4)

    image = cv2.imread('../../data/nit_sqr.jpg')
    pnts2 = h.read_image(image, 20)
    hom2 = h.calculate(pnts2)
    print pnts2
    print hom2
    print np.around(h.transform(hom2, pnts2), decimals=4)
