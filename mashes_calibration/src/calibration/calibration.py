#!/usr/bin/env python

import cv2
import numpy as np
from utils import Utils
from homography import Homography
from tcp_to_cam import TCP_to_cam


if __name__ == '__main__':

    process = Utils()
    scale = 8
    image_original = cv2.imread('../../data/nit_sqr.jpg')
    h = Homography()
    points = h.read_image(image_original, scale)
    hom = h.calculate(points)
    s = h.scale(hom)
    cam_H_cal = hom

    print '# Camera Calibration'
    print 'Homography:', cam_H_cal

    image_rotate_1 = cv2.imread('pics/pose1.jpg')
    image_rotate_2 = cv2.imread('pics/pose2.jpg')
    image_rotate_3 = cv2.imread('pics/pose3.jpg')
    image_translate_1 = cv2.imread('pics/pose4.jpg')
    (size_x, size_y, colors) = image_rotate_1.shape
    tcp = TCP_to_cam(image_rotate_1, image_rotate_2, image_rotate_3, image_translate_1, scale)
    pb_1, pb_2, pb_3, pb_4 = tcp.calculate_perp_bisector()
    xc, yc = tcp.calculate_intersection(pb_1, pb_2, pb_3, pb_4, size_x, size_y)

    x_real = xc*s
    y_real = yc*s
    print x_real, "mm, ", y_real, "mm from image center"

#--------------------------------------------------------------------#
    a = tcp.calculate_translation()
    TCP_H_cam = tcp.calculate_matrix(xc, yc, a)
    print TCP_H_cam


    #--------------------------------------------------------------------#
    TCP_H_cal = TCP_H_cam * cam_H_cal
    print TCP_H_cal

    # cal_h_base = [[ , , ], [ , , ], [ , , ]]
    # tcp_h_base = tcp_h_cal * cal_h_base
