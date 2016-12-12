import cv2
import numpy as np
from scipy import linalg

from homography import Homography

BLUE = [255, 0, 0]
RED = [0, 255, 0]
GREEN = [0, 0, 255]


class Representation():
    def __init__(self):
        pass

    def transform(self, hom, pnts):
        pnts = np.float32([
            np.dot(hom, np.array([pnt[0], pnt[1], 1])) for pnt in pnts])
        pnts = np.float32([pnt / pnt[2] for pnt in pnts])
        return pnts[:, :2]

    def define_camera(self, image, h):
        im_measures = cv2.warpPerspective(image, h, (500, 500))
        return im_measures

    def draw_axis_camera(self, image, pnts):
        cv2.line(image, (pnts[0][0], pnts[0][1]), (pnts[1][0], pnts[1][1]), RED, 2)
        cv2.line(image, (pnts[0][0], pnts[0][1]), (pnts[2][0], pnts[2][1]), GREEN, 2)
        return image

    def draw_points(self, image, pnts):
        for pnt in pnts:
            cv2.circle(image, (int(pnt[0]), int(pnt[1])), 10, BLUE, -1)
        return image


if __name__ == '__main__':

    pnts_pattern = np.float32([[0, 0], [1.1, 0], [0, 1.1], [1.1, 1.1]])
    pnts_image = np.float32([[579.3, 118.0], [572.7, 276.0], [425.3, 108.7], [412.7, 262.0]])

    homography = Homography(pnts_pattern)
    homography_uEye = homography.calculate(pnts_image)

    im_ueye = cv2.imread('../../data/calibration/vis/frame07.jpg')

    r1 = Representation()

    pnts = np.float32([[0, 0], [1, 0], [0, 1]])
    corners = np.float32([[-2.5, -2.5], [2.5, -2.5], [-2.5, 2.5], [2.5, 2.5]])
    pnts_final = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])

    print "uEye Homography: "
    print homography_uEye
    inv_homography_uEye = linalg.inv(homography_uEye)

    pxl_TCP_uEye = np.float32([[320, 256]])
    pnts_TCP_uEye = r1.transform(homography_uEye, pxl_TCP_uEye)[0]
    print "TCP origin: ", pnts_TCP_uEye

    a = np.deg2rad(0)
    Frame_uEye = np.float32([[np.cos(a),  -np.sin(a), pnts_TCP_uEye[0]],
                             [np.sin(a),  np.cos(a), pnts_TCP_uEye[1]],
                             [0, 0, 1]])
    inv_Frame_uEye = linalg.inv(Frame_uEye)

    pxls_uEye_camera = r1.transform(inv_homography_uEye, pnts)
    print "Pixels uEye", pxls_uEye_camera
#------------------
    im_measures_ueye = r1.define_camera(im_ueye.copy(), homography_uEye)
    #cv2.imshow("Warped Measures Image ueye", im_measures_ueye)
#------------------ Data in (c)mm
    image_axis_ueye = r1.draw_axis_camera(im_measures_ueye, pnts)
    #cv2.imshow("Axis Measures Image ueye", image_axis_ueye)
#------------------
    pnts_Frame_uEye = pnts
    pnts_uEye = r1.transform(Frame_uEye, pnts_Frame_uEye)
    print 'Points uEye', pnts_uEye
    image_result_ueye = r1.draw_axis_camera(image_axis_ueye, pnts_uEye)
#------------------
    corners_Frame_uEye = corners
    corners_uEye = r1.transform(Frame_uEye, corners_Frame_uEye)
    print 'Corners uEye', corners_uEye
    img = r1.draw_points(image_result_ueye, corners_uEye)
    #cv2.imshow("Image ueye", img)
#------------------
    pxls_uEye = r1.transform(inv_homography_uEye, corners_uEye)
    print "Pixels corners uEye", pxls_uEye

    hom_final_uEye, status = cv2.findHomography(pxls_uEye.copy(), pnts_final)
    print hom_final_uEye
    img_final_uEye = r1.define_camera(im_ueye, hom_final_uEye)
    #cv2.imshow("Image uEye final", img_final_uEye)

#------------------Visualize Axis pattern
    pnts_axis_pattern = pnts
    pxls_axis_pattern = r1.transform(inv_homography_uEye, pnts_axis_pattern)
    pnt_axis_pattern_final = r1.transform(hom_final_uEye, pxls_axis_pattern)
    img_pattern_uEye = r1.draw_axis_camera(img_final_uEye, pnt_axis_pattern_final)

#------------------Visualize Axis TCP
    pnts_axis = pnts
    pnts_axis_TCP = r1.transform(Frame_uEye, pnts_axis)
    pxls_axis_TCP = r1.transform(inv_homography_uEye, pnts_axis_TCP)
    print "Pixels axis TCP uEye", pxls_axis_TCP
    pnt_axis_TCP_final = r1.transform(hom_final_uEye, pxls_axis_TCP)
    print 'Points axis TCP final', pnt_axis_TCP_final
    img_final_axis_uEye = r1.draw_axis_camera(img_pattern_uEye, pnt_axis_TCP_final)
    cv2.imshow("Image uEye axis TCP", img_final_axis_uEye)

#------------------
#------------------
#------------------
#------------------

    im_NIT = cv2.imread('../../data/calibration/nit/frame7.jpg')
    r2 = Representation()

    #pxls_pattern_NIT = np.float32([[7.77, 16.29], [10.56, 16.14], [7.65, 18.96], [10.65, 18.9]])
    pxls_pattern_NIT = np.float32([[8, 15.4], [8, 18.3], [10.7, 15.4], [10.7, 18.3]])

    homography_NIT = homography.calculate(pxls_pattern_NIT)
    print "NIT Homography: "
    print homography_NIT
    inv_homography_NIT = linalg.inv(homography_NIT)

    pxl_TCP_NIT = np.float32([[13.5, 19]])
    pnts_TCP_NIT = r2.transform(homography_NIT, pxl_TCP_NIT)[0]
    print pnts_TCP_NIT

    a = np.deg2rad(0)
    Frame_NIT = np.float32([[np.cos(a),  -np.sin(a), pnts_TCP_NIT[0]],
                            [np.sin(a),  np.cos(a), pnts_TCP_NIT[1]],
                            [0, 0, 1]])

    inv_Frame_NIT = linalg.inv(Frame_NIT)


    pxls_NIT_camera = r2.transform(inv_homography_NIT, pnts)
    print "Pixels NIT", pxls_NIT_camera
#------------------
    im_measures_NIT = r2.define_camera(im_NIT.copy(), homography_NIT)
    #cv2.imshow("Warped Measures Image NIT", im_measures_NIT)
#------------------
    image_axis_NIT = r2.draw_axis_camera(im_measures_NIT, pnts)
    #cv2.imshow("Axis Measures Image NIT", image_axis_NIT)
#------------------
    pnts_Frame_NIT = pnts
    pnts_NIT = r2.transform(Frame_NIT, pnts_Frame_NIT)
    print 'Points NIT', pnts
    image_result_NIT = r2.draw_axis_camera(image_axis_NIT, pnts_NIT)
    #cv2.imshow("Axis Measures and TCP Image NIT", image_result_NIT)
#------------------
    corners_Frame_NIT = corners
    corners_NIT = r2.transform(Frame_NIT, corners_Frame_NIT)
    print 'Corners NIT', corners_NIT
    img = r2.draw_points(image_result_NIT, corners_NIT)
    #cv2.imshow("Image NIT", img)
#------------------
    pxls_NIT = r2.transform(inv_homography_NIT, corners_NIT)
    print "Pixels corners NIT", pxls_NIT

    hom_final_NIT, status = cv2.findHomography(pxls_NIT.copy(), pnts_final)
    print hom_final_NIT

    img_final_NIT = r2.define_camera(im_NIT, hom_final_NIT)
    #cv2.imshow("Image NIT final", img_final_NIT)

#------------------Visualize Axis pattern
    pnts_axis_pattern = pnts
    pxls_axis_pattern = r2.transform(inv_homography_NIT, pnts_axis_pattern)
    pnt_axis_pattern_final = r2.transform(hom_final_NIT, pxls_axis_pattern)
    img_pattern_NIT = r2.draw_axis_camera(img_final_NIT, pnt_axis_pattern_final)

#------------------Visualize Axis TCP
    pnts_axis = pnts
    pnts_axis_TCP = r2.transform(Frame_NIT, pnts_axis)
    pxls_axis_TCP = r2.transform(inv_homography_NIT, pnts_axis_TCP)
    print "Pixels axis TCP NIT", pxls_axis_TCP
    pnt_axis_TCP_final = r2.transform(hom_final_NIT, pxls_axis_TCP)
    print 'Points axis TCP final', pnt_axis_TCP_final
    img_final_axis_NIT = r2.draw_axis_camera(img_pattern_NIT, pnt_axis_TCP_final)
    cv2.imshow("Image NIT axis TCP", img_final_axis_NIT)

#------------------
#------------------
    img_final = cv2.addWeighted(img_final_axis_uEye, 0.2, img_final_axis_NIT, 0.8, 0)
    cv2.imshow("Image final", img_final)

    cv2.waitKey(0)
