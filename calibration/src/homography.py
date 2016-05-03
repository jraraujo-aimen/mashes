#!/usr/bin/env python

import cv2
import numpy as np
from utils import Utils


class Homography():
    def __init__(self, pts_pattern, im_ptr, scale):
        self.process = Utils()
        self.pts_pattern = np.vstack(pts_pattern).astype(float)

        # Read source image.
        self.pts_ptr= self.read_image(im_ptr, scale)


    def read_image(self, image, scale):
        # Read source image.
        (size_x, size_y, colors) = image.shape
        image_resized = cv2.resize(image, (size_x*scale, size_y*scale), interpolation=cv2.INTER_NEAREST)
        # Show image and wait for 2 clicks.
        cv2.imshow("Image", image_resized)
        pts_image_resized = self.process.points_average(image_resized)
        pts_image = pts_image_resized/scale
        return pts_image

    def calculate_homography(self):
        # Calculate the homography
        cam_h_cal, status_1 = cv2.findHomography(self.pts_ptr, self.pts_pattern)
        s = self.scale(cam_h_cal)
        return cam_h_cal, s

    def scale(self, cam_h_cal):
        s_0 = cam_h_cal[0, 0]
        s_1 = cam_h_cal[1, 1]
        s = np.mean([s_0, s_1])
        return s


if __name__ == '__main__':
    pts_font = [[0, 0], [6, 0], [0, 6], [6, 6]]
    image = cv2.imread('pics/nit_sqr.jpg')
    h = Homography(pts_font, image, 8)
    cam_H_cal, s= h.calculate_homography()
    print cam_H_cal, s
