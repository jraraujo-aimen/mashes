#!/usr/bin/env python

import cv2
import math
import numpy as np
from utils import Utils
from bisector import Perp_bisector, Intersection
from translation import Translation


class ScaleCamera(object):
    def __init__(self):
        self.process = Utils()
        self.pts_ptr = []
        self.pts_ptr_moved = []

    def homography(self, im_src, pts_font, im_ptr):
        # Show image and wait for 4 clicks.
        self.pts_ptr = self.process.points_average(im_ptr)
        # Calculate the homography
        h_1, status_1 = cv2.findHomography(pts_font, self.pts_ptr)

        # Calculate the homography
        self.cam_H_cal, status_1 = cv2.findHomography(self.pts_ptr, pts_font)
        print "cam_H_cal"
        print self.cam_H_cal
        print "\n"

        # # Warp source image to destination based on homography
        # im_out = cv2.warpPerspective(im_src, h_1, (im_ptr.shape[1], im_ptr.shape[0]))
        # # Display images
        # cv2.imshow("Warped Source Image", im_out)
        # cv2.waitKey(0)
        return self.cam_H_cal

    def offset_TCP(self, im_ptr_rotate):

        # Show image and wait for 4 clicks.
        self.pts_ptr_rotate = self.process.points_average(im_ptr_rotate)

        bisector = []
        for i in range(0, len(self.pts_ptr)):
            bisector_i = Perp_bisector(self.pts_ptr[i], self.pts_ptr_rotate[i])
            bisector.append(bisector_i)


        i_12 = Intersection(bisector[0], bisector[1])
        i_14 = Intersection(bisector[0], bisector[3])
        i_23 = Intersection(bisector[1], bisector[2])
        i_34 = Intersection(bisector[2], bisector[3])
        int_x = [i_12.x, i_14.x, i_23.x, i_34.x]
        int_y = [i_12.y, i_14.y, i_23.y, i_34.y]

        self.xc = np.mean(int_x)
        self.yc = np.mean(int_y)
        print "Offset of tool axis from camera reference system"
        print self.xc, self.yc
        print "\n"

    def turn_TCP(self, im_ptr_moved):

        # Show image and wait for 4 clicks.
        self.pts_ptr_moved = self.process.points_average(im_ptr_moved)

        ds = []
        for i in range(0, len(self.pts_ptr_rotate)):
            ds_i = Translation(self.pts_ptr_rotate[i], self.pts_ptr_moved[i])
            ds.append(ds_i)

        d_x = []
        d_y = []
        angle = []
        for i in range(0, len(ds)):
            d_x.append(ds[i].dx)
            d_y.append(ds[i].dy)
            angle.append(math.atan((-1)*ds[i].dy/ds[i].dx))

        x = np.mean(d_x)
        y = np.mean(d_y)
        self.a = np.mean(angle)
        print "Tool Axis movement over x-y plane"
        print x, y
        print "Turn of tool axis from camera reference system"
        print math.degrees(self.a)
        print "\n"
        return self.homogeneous_transf()

    def homogeneous_transf(self):
        self.tool_H_cam = [[math.cos(self.a), (-1)*math.sin(self.a), self.xc], [math.sin(self.a), math.cos(self.a), self.yc], [0, 0, 1]]
        print self.tool_H_cam




if __name__ == '__main__':
    scale = ScaleCamera()

    pts_font = [[0, 0], [190, 0], [0, 190], [190, 190]]
    pts_font = np.vstack(pts_font).astype(float)
    im_src = cv2.imread('pattern_trans_2.png')
    # Read source image.
    im_ptr = cv2.imread('pattern_2.png')
    scale.homography(pts_font, im_ptr)

    # Read source image.
    im_ptr_rotate = cv2.imread('pattern_rotate.png')
    scale.offset_TCP(im_ptr_rotate)

    # Read source image.
    im_ptr_moved = cv2.imread('pattern_moved.png')
    scale.turn_TCP(im_ptr_moved)
