#!/usr/bin/env python

import cv2
import math
import itertools
import numpy as np
from utils import Utils
from bisector import Perp_bisector, Intersection
from translation import Translation


class TCP():
    def __init__(self):
        self.process = Utils()

    def read_image(self, image, scale):
        # Read source image.
        (size_y, size_x, colors) = image.shape
        image_resized = cv2.resize(image, (size_x*scale, size_y*scale), interpolation=cv2.INTER_NEAREST)
        # Show image and wait for 2 clicks.
        cv2.imshow("Image", image_resized)
        pts_image_resized = self.process.points_average(image_resized, 1)
        pts_image = pts_image_resized/scale
        return pts_image

    def calculate_perp_bisector(self, pxl_pnts):
        pb = []
        results = itertools.combinations(range(len(pxl_pnts)), 2)
        # convert the combination iterator into a numpy array
        combs = np.array(list(results))
        for k in combs:
            pb.append(Perp_bisector(pxl_pnts[k[0]][0], pxl_pnts[k[1]][0]))
        return pb

    def calculate_intersection(self, pbs):
        intc = []
        results = itertools.combinations(range(len(pbs)), 2)
        # convert the combination iterator into a numpy array
        combs = np.array(list(results))
        for k in combs:
            intc.append(Intersection(pbs[k[0]], pbs[k[1]]))
        int_x = []
        int_y = []
        for i in intc:
            int_x.append(i.x)
            int_y.append(i.y)
        # print "Intersection points: "
        # print 'X:', int_x
        # print 'Y:', int_y
        # print " "
        x = np.mean(int_x)
        y = np.mean(int_y)
        return x, y

    def calculate_origin(self, pxl_pnts):
        pbs = self.calculate_perp_bisector(pxl_pnts)
        xc, yc = self.calculate_intersection(pbs)
        pxl_origin = np.float32([[xc, yc]])
        return pxl_origin


    def calculate_translation_x(self, pnts1, pnts3, dist):
        ds = []
        for i in range(0, len(pnts1)):
            ds_i = Translation(pnts1[i], pnts3[i])
            ds.append(ds_i)
        d_x = []
        d_y = []
        angle = []
        s = []
        for i in range(0, len(ds)):
            d_x.append(ds[i].dx)
            d_y.append(ds[i].dy)
            if ds[i].dx < 0 and ds[i].dy < 0:
                angle.append(math.atan(ds[i].dx/ds[i].dy) + math.pi/2)
            elif ds[i].dx < 0 and ds[i].dy > 0:
                angle.append(math.atan(ds[i].dx/ds[i].dy) - math.pi/2)
            else:
                angle.append(math.atan((-1)*ds[i].dy/ds[i].dx))
            s.append(ds[i].s)
        a = np.mean(angle)
        s = np.mean(s)
        factor = dist/s
        print "Scale factor:", factor, "Angle:", math.degrees(a)
        return a, factor

    def calculate_translation_y(self, pnts1, pnts2, dist):
        ds = []
        for i in range(0, len(pnts1)):
            ds_i = Translation(pnts1[i], pnts2[i])
            ds.append(ds_i)
        d_x = []
        d_y = []
        angle = []
        s = []
        for i in range(0, len(ds)):
            d_x.append(ds[i].dx)
            d_y.append(ds[i].dy)
            if ds[i].dx > 0 and ds[i].dy < 0:
                angle.append(math.atan((-1)*ds[i].dy/ds[i].dx) + math.pi/2)
            elif ds[i].dx < 0 and ds[i].dy < 0:
                angle.append(math.atan((-1)*ds[i].dy/ds[i].dx) - math.pi/2)
            else:
                angle.append(math.atan(ds[i].dx/ds[i].dy))
            s.append(ds[i].s)
        a = np.mean(angle)
        s = np.mean(s)
        factor = dist/s
        print "Scale factor:", factor, "Angle:", math.degrees(a)
        return a, factor

    def calculate_orientation(self, pnts_origin, pnts_x, pnts_y, d_x, d_y):
        angle_y, factor_a = self.calculate_translation_y(pnts_origin, pnts_y, d_y)
        angle_x, factor_b = self.calculate_translation_x(pnts_origin, pnts_x, d_x)
        factor = np.mean([factor_a, factor_b])
        print " "
        a_y = math.degrees(angle_y)
        a_x = math.degrees(angle_x)
        if -5 < (a_y - a_x) < 5:
            angle_y = np.mean([angle_y, angle_x])
            angle_x = angle_y
        return factor, angle_y, angle_x

    def calculate_matrix(self, xc, yc, a):
        tcp_h_cam = [[math.cos(a), (-1)*math.sin(a), xc], [math.sin(a), math.cos(a), yc], [0, 0, 1]]
        return tcp_h_cam

    def transform(self, hom, pnts):
        pnts = np.float32([
            np.dot(hom, np.array([pnt[0], pnt[1], 1])) for pnt in pnts])
        pnts = np.float32([pnt / pnt[2] for pnt in pnts])
        return pnts[:, :2]

if __name__ == '__main__':

    tcp_uEye = TCP()

    #---- Origin -----#
    pxl_pnts1 = np.float32([[536, 358]])
    pxl_pnts2 = np.float32([[509, 321]])
    pxl_pnts3 = np.float32([[485, 280]])
    pxl_pnts4 = np.float32([[432, 244]])
    pxl_pnts5 = np.float32([[365, 230]])
    pxl_pnts6 = np.float32([[319, 247]])
    pxl_pnts7 = np.float32([[299, 252]])
    pxl_pnts8 = np.float32([[282, 261]])
    pxl_pnts9 = np.float32([[260, 276]])
    pxl_pnts = [pxl_pnts1, pxl_pnts2, pxl_pnts3, pxl_pnts4, pxl_pnts5, pxl_pnts6, pxl_pnts7, pxl_pnts8, pxl_pnts9]
    #---- Orientation -----#
    pxl_pnts_origin = np.float32([[497, 195]])
    pxl_pnts_x = np.float32([[510, 457]])
    pxl_pnts_y = np.float32([[98, 199]])

    pxl_TCP_uEye = tcp_uEye.calculate_origin(pxl_pnts)
    print "TCP uEye:", pxl_TCP_uEye

    factor_uEye, angle_y_uEye, angle_x_uEye = tcp_uEye.calculate_orientation(pxl_pnts_origin, pxl_pnts_x, pxl_pnts_y, 2, 3)
    # factor = np.mean([factor_a, factor_b])
    # angle = np.mean([a, b])
    # print " "


    tcp_NIT = TCP()

    #---- Origin -----#
    pxl_pnts1 = np.float32([[9.02, 21.04]])
    pxl_pnts2 = np.float32([[9.105, 20.58]])
    pxl_pnts3 = np.float32([[9.315, 19.125]])
    pxl_pnts4 = np.float32([[11.17, 18.66]])
    pxl_pnts5 = np.float32([[12.48, 18.54]])
    pxl_pnts6 = np.float32([[13.05, 18.30]])
    pxl_pnts7 = np.float32([[13.51, 18.39]])
    pxl_pnts8 = np.float32([[13.995, 18.825]])
    pxl_pnts9 = np.float32([[13.80, 19.28]])
    pxl_pnts = [pxl_pnts1, pxl_pnts2, pxl_pnts3, pxl_pnts4, pxl_pnts5, pxl_pnts6, pxl_pnts7, pxl_pnts8, pxl_pnts9]
    #---- Orientation -----#
    pxl_pnts_origin = np.float32([[9.645, 17.715]])
    pxl_pnts_y = np.float32([[17.475, 17.505]])
    pxl_pnts_x = np.float32([[9.42, 22.88]])

    pxl_TCP_NIT = tcp_NIT.calculate_origin(pxl_pnts)
    print "TCP NIT:", pxl_TCP_NIT

    factor_NIT, angle_y_NIT, angle_x_NIT = tcp_NIT.calculate_orientation(pxl_pnts_origin, pxl_pnts_x, pxl_pnts_y, 2, 3)
