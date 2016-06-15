
from calibration_camera import Cal_camera
import numpy as np
import glob
import cv2


if __name__ == '__main__':
#-------------------------------------------------------#
    pnts = np.float32([[0, 0],
                       [1, 0],
                       [0, 1]])

    corners = np.float32([[-3.2, -3.2],
                          [3.2, -3.2],
                          [-3.2, 3.2],
                          [3.2, 3.2]])

    pnts_final = np.float32([[0, 0],
                             [500, 0],
                             [0, 500],
                             [500, 500]])

    pnts_pattern = np.float32([[0, 0],
                               [1.1, 0],
                               [0, 1.1],
                               [1.1, 1.1]])

    uEye = Cal_camera('uEye', pnts_pattern)
    im = cv2.imread("../../data/calibration/vis/move_o.jpg")
    pxls_pattern_uEye = uEye.get_pxls_homography(im)
    print pxls_pattern_uEye
    uEye.calculate_homography(pxls_pattern_uEye)

    #---- Origin -----#
    files = glob.glob("../../data/calibration/vis/frame*.jpg")
    pxl_pnts = uEye.get_pxl_origin(files)
    #---- Orientation -----#
    files_move = glob.glob("../../data/calibration/vis/move*.jpg")
    pxl_pnts_origin, pxl_pnts_x, pxl_pnts_y = uEye.get_pxl_orientation(files_move)
    pxl_TCP, factor, angle_y, angle_x = uEye.calculate_TCP_orientarion(pxl_pnts, pxl_pnts_origin, pxl_pnts_x, pxl_pnts_y, 2, 3)

    angle = uEye.calculate_angle_TCP(pxl_pnts_origin, pxl_pnts_x, pxls_pattern_uEye)
    print "Angle TCP uEye: ", angle
    uEye.calculate_frame(pxl_TCP, angle)

    hom_final_uEye = uEye.calculate_hom_final(im, pnts, corners, pnts_final)
    print "uEye parameters: "
    uEye.visualize_data()
    uEye.write_config_file()

    files = glob.glob("../../data/calibration/vis/frame*.jpg")
    for f in sorted(files):
        im_uEye = cv2.imread(f)
        img_final_uEye = uEye.rep.define_camera(im_uEye, hom_final_uEye)
        img_final_uEye_axis = uEye.draw_axis(pnts, img_final_uEye)
        cv2.imshow("Image: ", img_final_uEye_axis)
        cv2.waitKey(0)

#-------------------------------------------------------------#
    scale = 30
    NIT = Cal_camera('NIT', pnts_pattern)
    im = cv2.imread("../../data/calibration/nit/move_o.jpg")
    pxls_pattern_NIT = NIT.get_pxls_homography(im, scale)
    print pxls_pattern_NIT
    NIT.calculate_homography(pxls_pattern_NIT)

    #---- Origin -----#
    files = glob.glob("../../data/calibration/nit/frame*.jpg")
    pxl_pnts = NIT.get_pxl_origin(files, scale)
    #---- Orientation -----#
    files_move = glob.glob("../../data/calibration/nit/move*.jpg")
    pxl_pnts_origin, pxl_pnts_x, pxl_pnts_y = NIT.get_pxl_orientation(files_move, scale)
    pxl_TCP, factor, angle_y, angle_x = NIT.calculate_TCP_orientarion(pxl_pnts, pxl_pnts_origin, pxl_pnts_x, pxl_pnts_y, 2, 3)

    angle = NIT.calculate_angle_TCP(pxl_pnts_origin, pxl_pnts_x, pxls_pattern_NIT)
    print "Angle TCP NIT: ", angle
    NIT.calculate_frame(pxl_TCP, angle)

    hom_final_NIT = NIT.calculate_hom_final(im, pnts, corners, pnts_final)
    print "NIT parameters: "
    NIT.visualize_data()
    NIT.write_config_file()

    files = glob.glob("../../data/calibration/nit/frame*.jpg")
    for f in sorted(files):
        im_NIT = cv2.imread(f)
        img_final_NIT = NIT.rep.define_camera(im_NIT, hom_final_NIT)
        img_final_NIT_axis = NIT.draw_axis(pnts, img_final_NIT)
        cv2.imshow("Image: ", img_final_NIT)
        cv2.waitKey(0)
