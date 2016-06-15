import yaml
import cv2
import glob
import numpy as np
from scipy import linalg

BLUE = [255, 0, 0]
RED = [0, 255, 0]
GREEN = [0, 0, 255]

pnts = np.float32([[0, 0],
                   [1, 0],
                   [0, 1]])


class Projection():
    def __init__(self):
        #self.hom = np.zeros(3, 3)
        print ""

    def load_configuration(self, filename):
        with open(filename, 'r') as f:
            data = yaml.load(f)
        self.hom = np.array(data['hom'])
        self.inv_hom = np.array(data['inv_hom'])
        self.hom_vis = np.array(data['hom_vis'])
        self.inv_hom_vis = linalg.inv(self.Frame)

    def project_image(self, image, h):
        im_measures = cv2.warpPerspective(image, h, (500, 500))
        return im_measures

    def transform(self, hom, pnts):
        pnts = np.float32([
            np.dot(hom, np.array([pnt[0], pnt[1], 1])) for pnt in pnts])
        pnts = np.float32([pnt / pnt[2] for pnt in pnts])
        return pnts[:, :2]

    def draw_axis_camera(self, image, pnts):
        cv2.line(image, (pnts[0][0], pnts[0][1]), (pnts[1][0], pnts[1][1]), RED, 2)
        cv2.line(image, (pnts[0][0], pnts[0][1]), (pnts[2][0], pnts[2][1]), GREEN, 2)
        return image

    def draw_TCP_axis(self, pnts, img):
        pxls_TCP = self.transform(self.hom, pnts)
        pnt_TCP_final = self.transform(self.hom_vis, pxls_TCP)
        img_final_axis = self.draw_axis_camera(img, pnt_TCP_final)
        return img_final_axis

    def draw_point(self, image, pnts):
        pxls = self.transform(self.hom, pnts)
        pnts_final = self.transform(self.hom_vis, pxls)
        for pnt in pnts_final:
            cv2.circle(image, (int(pnt[0]), int(pnt[1])), 10, BLUE, -1)
        return image

if __name__ == '__main__':

    pnt = np.float32([[0, 0]])
    p_uEye = Projection()
    p_uEye.load_configuration('../config/uEye_config.yaml')
    images_uEye = []
    files_uEye = glob.glob("../data/calibration/vis/frame*.jpg")
    for f in sorted(files_uEye):
        im_uEye = cv2.imread(f)
        im_uEye_f1 = p_uEye.project_image(im_uEye, p_uEye.hom_vis)
        im_uEye_f2 = p_uEye.draw_TCP_axis(pnts, im_uEye_f1)
        im_uEye_f3 = p_uEye.draw_point(im_uEye_f2, pnt)
        images_uEye.append(im_uEye_f3)
        cv2.imshow("Image: ", im_uEye_f3)
        cv2.waitKey(0)

    p_NIT = Projection()
    p_NIT.load_configuration('../config/NIT_config.yaml')
    images_NIT = []
    files_NIT = glob.glob("../data/calibration/nit/frame*.jpg")
    for f in sorted(files_NIT):
        im_NIT = cv2.imread(f)
        im_NIT_f1 = p_NIT.project_image(im_NIT, p_NIT.hom_vis)
        im_NIT_f2 = p_NIT.draw_TCP_axis(pnts, im_NIT_f1)
        im_NIT_f3 = p_uEye.draw_point(im_NIT_f2, pnt)
        images_NIT.append(im_NIT_f3)
        cv2.imshow("Image: ", im_NIT_f3)
        cv2.waitKey(0)

    for im_NIT, im_uEye in zip(images_NIT, images_uEye):
        im_final = cv2.addWeighted(im_uEye, 0.4, im_NIT, 0.6, 0)
        cv2.imshow("Image final", im_final)
        cv2.waitKey(0)
