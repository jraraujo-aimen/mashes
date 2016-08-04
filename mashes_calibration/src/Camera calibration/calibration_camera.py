import cv2
import yaml
import numpy as np
from os.path import basename
from scipy import linalg
from tcp import TCP
from homography import Homography
from represent import Representation
from angle import Angle


class Cal_camera():
    def __init__(self, camera, pnts_pattern):
        self.rep = Representation()
        self.h = Homography(pnts_pattern)
        self.tcp = TCP()
        self.a = Angle()
        self.camera = camera
        self.hom = np.zeros((3, 3))
        self.inv_hom = np.zeros((3, 3))
        self.Frame = np.zeros((3, 3))
        self.inv_Frame = np.zeros((3, 3))
        self.hom_final_camera = np.zeros((3, 3))

    def get_pxls_homography(self, image, scale=1):
        pts_image = self.h.read_image(image, scale)
        return pts_image

    def calculate_homography(self, pxls_pattern):
        self.hom = self.h.calculate(pxls_pattern)
        self.inv_hom = linalg.inv(self.hom)

    def get_pxl_origin(self, folder, scale=1):
        pxl_pnts = []
        for f in sorted(folder):
            im_uEye = cv2.imread(f)
            pxl_pnts.append(self.tcp.read_image(im_uEye, scale))
        return pxl_pnts

    def get_pxl_orientation(self, folder, scale=1):
        for f in sorted(folder):
            name = basename(f)
            if name == "move_x.jpg":
                im_uEye = cv2.imread(f)
                pxl_pnts_x = self.tcp.read_image(im_uEye, scale)
            elif name == "move_y.jpg":
                im_uEye = cv2.imread(f)
                pxl_pnts_y = self.tcp.read_image(im_uEye, scale)
            elif name == "move_o.jpg":
                im_uEye = cv2.imread(f)
                pxl_pnts_origin = self.tcp.read_image(im_uEye, scale)

        return pxl_pnts_origin, pxl_pnts_x, pxl_pnts_y

    def calculate_TCP_orientarion(self, pxl_pnts, pxl_pnts_origin, pxl_pnts_x, pxl_pnts_y, dx, dy):
        pxl_TCP = self.tcp.calculate_origin(pxl_pnts)
        print "TCP :", pxl_TCP
        factor, angle_y, angle_x = self.tcp.calculate_orientation(pxl_pnts_origin, pxl_pnts_x, pxl_pnts_y, dx, dy)
        return pxl_TCP, factor, angle_y, angle_x

    def calculate_angle_TCP(self, origin, axis_x, pattern):

        pnt_origin = self.rep.transform(self.hom, origin)
        pnt_x = self.rep.transform(self.hom, axis_x)
        pnt_pattern = self.rep.transform(self.hom, pattern)

        l_1 = np.float32([pnt_origin[0], pnt_x[0]])
        l_2 = pnt_pattern[0:2]
        vd1 = self.a.director_vector(l_1[0], l_1[1])
        vd2 = self.a.director_vector(l_2[0], l_2[1])
        angle = self.a.calculate_angle(vd1, vd2)
        return angle

    def calculate_frame(self, pxl_TCP, angle):
        pnts_TCP = self.rep.transform(self.hom, pxl_TCP)[0]

        a = np.deg2rad(angle)
        self.Frame = np.float32([[np.cos(a),  -np.sin(a), pnts_TCP[0]],
                                [np.sin(a),  np.cos(a), pnts_TCP[1]],
                                [0, 0, 1]])

        self.Orientation = np.float32([[np.cos(a),  -np.sin(a), 0],
                                       [np.sin(a),  np.cos(a), 0],
                                       [0, 0, 1]])
        self.inv_Orientation = linalg.inv(self.Orientation)
        self.inv_Frame = linalg.inv(self.Frame)

    def calculate_hom_final(self, img, pnts, corners, pnts_final):
        # pxls_camera = self.rep.transform(self.inv_hom, pnts)
        im_measures = self.rep.define_camera(img.copy(), self.hom)
        #------------------ Data in (c)mm
        image_axis = self.rep.draw_axis_camera(im_measures, pnts)
        #------------------
        pnts_Frame = pnts
        pnts_camera = self.rep.transform(self.Frame, pnts_Frame)

        image_axis_tcp = self.rep.draw_axis_camera(image_axis, pnts_camera)
        #------------------
        corners_Frame = corners
        corners_camera = self.rep.transform(self.Frame, corners_Frame)
        img = self.rep.draw_points(image_axis_tcp, corners_camera)
        #------------------
        pxls_corner = self.rep.transform(self.inv_hom, corners_camera)
        self.hom_final_camera, status = cv2.findHomography(pxls_corner.copy(), pnts_final)
        self.write_config_file()
        return self.hom_final_camera

    def write_config_file(self):
        hom_vis = self.hom_final_camera
        hom_TCP = np.dot(self.inv_hom, self.Frame)
        inv_hom_TCP = np.dot(self.inv_Frame, self.hom)
        data = dict(
            hom_vis=hom_vis.tolist(),
            hom=hom_TCP.tolist(),
            inv_hom=inv_hom_TCP.tolist(),
            )
        filename = '../../config/' + self.camera + '_config.yaml'
        with open(filename, 'w') as outfile:
            yaml.dump(data, outfile)
            print data

    def visualize_data(self):
        print "Homography: "
        print self.hom

        print "Frame: "
        print self.Frame

        print "Final homography: "
        print self.hom_final_camera

    def draw_pattern_axis(self, pnts, img):
        pnts_axis_pattern = pnts
        pxls_axis_pattern = self.rep.transform(self.inv_hom, pnts_axis_pattern)
        pnt_axis_pattern_final = self.rep.transform(self.hom_final_camera, pxls_axis_pattern)
        img_pattern_NIT = self.rep.draw_axis_camera(img, pnt_axis_pattern_final)
        return img_pattern_NIT

    def draw_TCP_axis(self, pnts, img):
        pnts_axis = pnts
        pnts_axis_TCP = self.rep.transform(self.Frame, pnts_axis)
        pxls_axis_TCP = self.rep.transform(self.inv_hom, pnts_axis_TCP)
        pnt_axis_TCP_final = self.rep.transform(self.hom_final_camera, pxls_axis_TCP)
        img_final_axis = self.rep.draw_axis_camera(img, pnt_axis_TCP_final)
        return img_final_axis

    def draw_axis(self, pnts, img):
        img_TCP = self.draw_pattern_axis(pnts, img)
        img_final = self.draw_TCP_axis(pnts, img_TCP)
        return img_final

if __name__ == '__main__':

    pnts = np.float32([[0, 0],
                       [1, 0],
                       [0, 1]])


    corners = np.float32([[-3.2, -3.2],
                          [3.2, -3.2],
                          [-3.2, 3.2],
                          [3.2, 3.2]])

    # corners = np.float32([[-2.5, -2.5],
    #                       [2.5, -2.5],
    #                       [-2.5, 2.5],
    #                       [2.5, 2.5]])

    pnts_final = np.float32([[0, 0],
                             [500, 0],
                             [0, 500],
                             [500, 500]])

    pnts_pattern = np.float32([[0, 0], [1.1, 0], [0, 1.1], [1.1, 1.1]])

    uEye = Cal_camera('uEye', pnts_pattern)
    #pxls_pattern_uEye = np.float32([[425.3, 108.7], [579.3, 118.0], [412.7, 262.0], [572.7, 276.0]])
    #pxls_pattern_uEye = np.float32([[569, 141], [564, 270], [440, 131], [436, 261]])
    pxls_pattern_uEye = np.float32([[567, 142], [563, 261], [434, 138], [429, 256]])
    uEye.calculate_homography(pxls_pattern_uEye)

    pxl_pnt_origin = np.float32([[501.5, 196]])
    pxl_pnt_x = np.float32([[510, 453]])
    pxl_TCP_uEye = np.float32([[320, 256]])

    angle = uEye.calculate_angle_TCP(pxl_pnt_origin, pxl_pnt_x, pxls_pattern_uEye)
    print "Angle TCP uEye: ", angle

    uEye.calculate_frame(pxl_TCP_uEye, angle)

    im_ueye = cv2.imread("../../data/calibration/vis/move_o.jpg")
    hom_final_uEye = uEye.calculate_hom_final(im_ueye, pnts, corners, pnts_final)

    img_final_uEye = uEye.rep.define_camera(im_ueye, hom_final_uEye)
    print "uEye parameters: "
    uEye.visualize_data()
    img_final_uEye_axis = uEye.draw_axis(pnts, img_final_uEye)
    cv2.imshow("Image uEye final", img_final_uEye_axis)




#------------------
#------------------

    NIT = Cal_camera('NIT', pnts_pattern)
    #pxls_pattern_NIT = np.float32([[8.19, 15.75], [10.98, 15.72], [8.10, 19.08], [11.04, 19.05]])
    #pxls_pattern_NIT = np.float32([[7.32, 16.05], [7.38, 19.62], [11.04, 15.99], [11.13, 19.53]])
    pxls_pattern_NIT = np.float32([[7.86, 15.72], [7.86, 18.75], [10.83, 15.72], [10.86, 18.69]])
    NIT.calculate_homography(pxls_pattern_NIT)

    pxl_pnt_origin = np.float32([[9.555, 17.655]])
    pxl_pnt_x = np.float32([[9.28, 23.04]])
    pxl_TCP_NIT = np.float32([[13.77, 19.69]])

    angle = NIT.calculate_angle_TCP(pxl_pnt_origin, pxl_pnt_x, pxls_pattern_NIT)
    print "Angle TCP NIT: ", angle
    NIT.calculate_frame(pxl_TCP_NIT, angle)

    im_NIT = cv2.imread("../../data/calibration/nit/move_o.jpg")
    hom_final_NIT = NIT.calculate_hom_final(im_NIT, pnts, corners, pnts_final)

    img_final_NIT = NIT.rep.define_camera(im_NIT, hom_final_NIT)
    print "NIT parameters: "
    NIT.visualize_data()
    img_final_NIT_axis = NIT.draw_axis(pnts, img_final_NIT)
    cv2.imshow("Image NIT final", img_final_NIT_axis)



#------------------
#------------------
#------------------
#------------------

    img_final = cv2.addWeighted(img_final_uEye_axis, 0.3, img_final_NIT_axis, 0.7, 0)
    cv2.imshow("Image final", img_final)

    cv2.waitKey(0)
