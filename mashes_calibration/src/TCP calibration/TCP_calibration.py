#!/usr/bin/env python

import cv2
import numpy as np
from utils import Utils
from bisector import Perp_bisector, Intersection


class TCP_to_cam():
    def __init__(self):
        self.process = Utils()

    def read_image(self, image, scale):
        # Read source image.
        (size_x, size_y, colors) = image.shape
        image_resized = cv2.resize(image, (size_x*scale, size_y*scale), interpolation=cv2.INTER_NEAREST)
        # Show image and wait for 2 clicks.
        cv2.imshow("Image", image_resized)
        pts_image_resized = self.process.points_average(image_resized, 2)
        pts_image = pts_image_resized/scale
        return pts_image

    def calculate_perp_bisector(self, pnts1, pnts2, pnts3, pnts4, pnts5, pnts6, pnts7):
        pb = []
        for k in range(len(pnts1)):
            pb.append(Perp_bisector(pnts1[k], pnts3[k]))
            pb.append(Perp_bisector(pnts2[k], pnts4[k]))
            pb.append(Perp_bisector(pnts3[k], pnts5[k]))
            pb.append(Perp_bisector(pnts4[k], pnts6[k]))
            pb.append(Perp_bisector(pnts5[k], pnts7[k]))
            pb.append(Perp_bisector(pnts6[k], pnts7[k]))
        return pb

    def calculate_intersection(self, pb_1, pb_2, pb_3, pb_4, pb_5, pb_6):
        i_13 = Intersection(pb_1, pb_2)
        i_24 = Intersection(pb_3, pb_4)
        i_35 = Intersection(pb_3, pb_5)
        i_46 = Intersection(pb_4, pb_6)
        int_x = [i_13.x, i_24.x, i_35.x, i_46.x]
        int_y = [i_13.y, i_24.y, i_35.y, i_46.y]
        print 'X:', int_x
        print 'Y:', int_y
        x = np.mean(int_x)
        y = np.mean(int_y)
        return x, y


if __name__ == '__main__':
    import os
    from homography import Homography
    dirname = '../../data/'

    scale = 8
    tcp = TCP_to_cam()
    # img1 = cv2.imread(os.path.join(dirname, 'pose1.jpg'))
    # pnts1 = tcp.read_image(img1, scale)

    tcp0 = np.float32([[329, 288.9]])
    pnts1 = np.float32([[177.5, 394]])
    pnts2 = np.float32([[221, 386.5]])
    pnts3 = np.float32([[294.5, 376.5]])
    pnts4 = np.float32([[368, 354]])
    pnts5 = np.float32([[431.5, 322]])
    pnts6 = np.float32([[504.5, 290.5]])
    pnts7 = np.float32([[562, 250.5]])



    h = Homography()
    # hom = np.float32([[0.341, -0.002, -2.588],
    #                   [-0.002, 0.322, -2.407],
    #                   [0.000, 0.000, 1.000]])
    hom = np.float32([[0, -0.00814, 512*0.00814],
                      [0.00814, 0, 0],
                      [0, 0, 1.00000]])

    pnts1 = h.transform(hom, pnts1)
    pnts2 = h.transform(hom, pnts2)
    pnts3 = h.transform(hom, pnts3)
    pnts4 = h.transform(hom, pnts4)
    pnts5 = h.transform(hom, pnts5)
    pnts6 = h.transform(hom, pnts6)
    pnts7 = h.transform(hom, pnts7)

    print "Ptos:", pnts1, pnts2, pnts3, pnts4, pnts5,  pnts6, pnts7
    tcp0 = h.transform(hom, tcp0)
    print " "
    print "Ideal TCP:", tcp0
    print " "

    pbs = tcp.calculate_perp_bisector(pnts1, pnts2, pnts3, pnts4, pnts5, pnts6, pnts7)
    xc, yc = tcp.calculate_intersection(pbs[0], pbs[1], pbs[2], pbs[3], pbs[4], pbs[5])

    print "From pattern origin:", xc, "mm, ", yc, "mm"
    print "TCP deviation", tcp0[0, 0] - xc, tcp0[0, 1] - yc
    mov = tcp0[0, 0] - xc, tcp0[0, 1] - yc
