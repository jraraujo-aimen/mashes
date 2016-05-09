#!/usr/bin/env python

import cv2
import math
import numpy as np
from utils import Utils
from bisector import Perp_bisector, Intersection
from translation import Translation

class TCP_to_cam():
    def __init__(self):
        self.process = Utils()

    def read_image(self, image, scale):
        # Read source image.
        (size_x, size_y, colors) = image.shape
        image_resized = cv2.resize(image, (size_x*scale, size_y*scale),interpolation=cv2.INTER_NEAREST)
        # Show image and wait for 2 clicks.
        cv2.imshow("Image", image_resized)
        pts_image_resized = self.process.points_average_rotate(image_resized)
        pts_image = pts_image_resized/scale
        return pts_image

    def calculate_perp_bisector(self, pnts1, pnts2, pnts3, pnts4, pnts5):
        pb = []
        for k in range(len(pnts1)):
            pb.append(Perp_bisector(pnts1[k], pnts3[k]))
            pb.append(Perp_bisector(pnts2[k], pnts4[k]))
            pb.append(Perp_bisector(pnts3[k], pnts5[k]))
            pb.append(Perp_bisector(pnts1[k], pnts5[k]))
        return pb

    def calculate_intersection(self, pb_1, pb_2, pb_3, pb_4):
        i_13 = Intersection(pb_1, pb_2)
        i_24 = Intersection(pb_3, pb_4)
        # i_35 = Intersection(pb_3, pb_5)
        # i_46 = Intersection(pb_4, pb_6)
        int_x = [i_13.x, i_24.x]
        int_y = [i_13.y, i_24.y]
        print 'X:', int_x
        print 'Y:', int_y
        x = np.mean(int_x)
        y = np.mean(int_y)
        return x, y

    def calculate_translation(self, pnts1, pnts2):
        ds = []
        for i in range(0, len(pnts1)):
            ds_i = Translation(pnts1[i], pnts2[i])
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
        print "Movement:", x, ",", y, "Angle:", math.degrees(a)
        return a

    def calculate_matrix(self, xc, yc, a):
        tcp_h_cam = [[math.cos(a), (-1)*math.sin(a), xc], [math.sin(a), math.cos(a), yc], [0, 0, 1]]
        return tcp_h_cam


if __name__ == '__main__':
    import os
    from homography import Homography
    dirname = '../../data/'

    scale = 8
    tcp = TCP_to_cam()
    # img1 = cv2.imread(os.path.join(dirname, 'pose1.jpg'))
    # pnts1 = tcp.read_image(img1, scale)
    # img2 = cv2.imread(os.path.join(dirname, 'pose2.jpg'))
    # pnts2 = tcp.read_image(img2, scale)
    # img3 = cv2.imread(os.path.join(dirname, 'pose3.jpg'))
    # pnts3 = tcp.read_image(img3, scale)
    # img4 = cv2.imread(os.path.join(dirname, 'pose4.jpg'))
    # pnts4 = tcp.read_image(img4, scale)
    # print pnts1, pnts2, pnts3
    tcp0 = np.float32([[322.6, 290.2]])
    pnts1 = np.float32([[104.5, 254.5]])
    pnts2 = np.float32([[206, 277]])
    pnts3 = np.float32([[343.5, 285.5]])
    pnts4 = np.float32([[477, 286]])
    pnts5 = np.float32([[605, 264.5]])

    h = Homography()
    # hom = np.float32([[0.341, -0.002, -2.588],
    #                   [-0.002, 0.322, -2.407],
    #                   [0.000, 0.000, 1.000]])
    hom = np.float32([[0, -0.00814, 512*0.00814],
                      [0.00814, 0, 0],
                      [0, 0, 1.00000]])

    #hom = np.float32(np.eye(3))
    pnts1 = h.transform(hom, pnts1)
    pnts2 = h.transform(hom, pnts2)
    pnts3 = h.transform(hom, pnts3)
    pnts4 = h.transform(hom, pnts4)
    pnts5 = h.transform(hom, pnts5)

    print pnts1, pnts2, pnts3, pnts4, pnts5
    tcp0 = h.transform(hom, tcp0)
    print "Ideal TCP:", tcp0

    pbs = tcp.calculate_perp_bisector(pnts1, pnts2, pnts3, pnts4, pnts5)
    xc, yc = tcp.calculate_intersection(pbs[0], pbs[1], pbs[2], pbs[3])

    print "From pattern origin:", xc, "mm, ", yc, "mm"
    print "TCP deviation", tcp0[0, 0] - xc, tcp0[0, 1] - yc
    mov = tcp0[0, 0] - xc, tcp0[0, 1] - yc
    #
    # a = tcp.calculate_translation()
    # TCP_H_cam = tcp.calculate_matrix(xc, yc, a)
    # print TCP_H_cam
