#!/usr/bin/env python

import cv2
import math
import numpy as np
from utils import Utils
from bisector import Perp_bisector, Intersection
from translation import Translation

class TCP_to_cam():
    def __init__(self, im_ptr_1, im_ptr_2, im_ptr_3, im_ptt_4, scale):
        self.process = Utils()
        # Read source images.
        self.pts_ptr_1 = self.read_image(im_ptr_1, scale)
        print self.pts_ptr_1
        self.pts_ptr_2 = self.read_image(im_ptr_2, scale)
        print self.pts_ptr_2
        self.pts_ptr_3 = self.read_image(im_ptr_3, scale)
        print self.pts_ptr_3
        self.pts_ptt_4 = self.read_image(im_ptt_4, scale)
        print self.pts_ptt_4

    def read_image(self, image, scale):
        # Read source image.
        (size_x, size_y, colors) = image.shape
        image_resized = cv2.resize(image, (size_x*scale, size_y*scale),interpolation=cv2.INTER_NEAREST)
        # Show image and wait for 2 clicks.
        cv2.imshow("Image", image_resized)
        pts_image_resized = self.process.points_average_rotate(image_resized)
        pts_image = pts_image_resized/scale
        return pts_image


    def calculate_perp_bisector(self):
        pb_1 = Perp_bisector(self.pts_ptr_1[0], self.pts_ptr_2[0])
        pb_2 = Perp_bisector(self.pts_ptr_2[0], self.pts_ptr_3[0])
        pb_3 = Perp_bisector(self.pts_ptr_1[1], self.pts_ptr_2[1])
        pb_4 = Perp_bisector(self.pts_ptr_2[1], self.pts_ptr_3[1])
        return pb_1, pb_2, pb_3, pb_4

    def calculate_intersection(self,pb_1, pb_2, pb_3, pb_4, size_x, size_y):
        i_12 = Intersection(pb_1, pb_2)
        i_34 = Intersection(pb_3, pb_4)
        int_x = [i_12.x, i_34.x]
        int_y = [i_12.y, i_34.y]
        print int_x, int_y

        x = np.mean(int_x)
        y = np.mean(int_y)

        xc = x - size_x/2
        yc = y - size_y/2
        return xc, yc

    def calculate_translation (self):
        ds = []
        for i in range(0, len(self.pts_ptt_4)):
            ds_i = Translation(self.pts_ptr_1[i], self.pts_ptt_4[i])
            ds.append(ds_i)

        d_x = []
        d_y = []
        angle = []
        for i in range(0, len(ds)):
            d_x.append(ds[i].dx)
            d_y.append(ds[i].dy)
            angle.append(math.atan(ds[i].dy/ds[i].dx))
            # angle.append(math.atan((-1)*ds[i].dy/ds[i].dx)) Negative: cam_to_TCP
            # angle.append(math.atan((-1)*ds[i].dy/ds[i].dx)) Positive: TCP_to_cam
        x = np.mean(d_x)
        y = np.mean(d_y)
        a = np.mean(angle)
        print "Movement:", x, "," ,y, "Angle:", math.degrees(a)
        return a

    def calculate_matrix (self,xc, yc, a):
        tcp_h_cam = [[math.cos(a), (-1)*math.sin(a), xc], [math.sin(a), math.cos(a), yc], [0, 0, 1]]
        return tcp_h_cam


if __name__ == '__main__':

    scale =8
    image_rotate_1 = cv2.imread('pics/pose1.jpg')
    image_rotate_2 = cv2.imread('pics/pose2.jpg')
    image_rotate_3 = cv2.imread('pics/pose3.jpg')
    image_translate_1 = cv2.imread('pics/pose4.jpg')
    (size_x, size_y, colors) = image_rotate_1.shape

    tcp = TCP_to_cam(image_rotate_1, image_rotate_2, image_rotate_3,image_translate_1, scale)
    pb_1, pb_2, pb_3, pb_4 = tcp.calculate_perp_bisector()
    xc, yc = tcp.calculate_intersection(pb_1, pb_2, pb_3, pb_4, size_x, size_y)


    s = 0.375
    x_real = xc*s
    y_real = yc*s
    print "Form image centre:", x_real,"cm, ", y_real, "cm"

    a = tcp.calculate_translation()
    TCP_H_cam = tcp.calculate_matrix(xc, yc, a)
    print TCP_H_cam
