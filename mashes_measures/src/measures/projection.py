import cv2
import yaml
import glob
import numpy as np
from scipy import linalg


BLUE = [255, 0, 0]
RED = [0, 255, 0]
GREEN = [0, 0, 255]


class Projection():
    def __init__(self, config=None):
        self.hom = np.eye(3)
        self.hom_vis = np.eye(3)
        if config is not None:
            self.load_configuration(config)

    def load_configuration(self, filename):
        with open(filename, 'r') as f:
            data = yaml.load(f)
        self.hom = np.array(data['hom'])
        hom_vis = np.float32([self.hom[1], self.hom[0], self.hom[2]])
        hom_vis = 100 * hom_vis
        hom_vis[0, 2] = hom_vis[0, 2] + 250
        hom_vis[1, 2] = hom_vis[1, 2] + 250
        hom_vis[2, 2] = 1
        self.hom_vis = hom_vis

    def project_image(self, image):
        im_measures = cv2.warpPerspective(image, self.hom_vis, (500, 500))
        return im_measures

    def transform(self, hom, pnts):
        pnts = np.float32([
            np.dot(hom, np.array([pnt[0], pnt[1], 1])) for pnt in pnts])
        pnts = np.float32([pnt / pnt[2] for pnt in pnts])
        return pnts[:, :2]

    def transform_ellipse(self, center, axis, angle):
        major_u = axis[0] / 2 * np.cos(angle) + center[0]
        major_v = axis[0] / 2 * np.sin(angle) + center[1]
        minor_u = axis[1] / 2 * np.cos(angle) + center[0]
        minor_v = axis[1] / 2 * np.sin(angle) + center[1]
        center = self.transform(self.hom, [center])[0]
        axis = self.transform(self.hom, np.float32([[major_u, major_v],
                                                    [minor_u, minor_v]]))
        axis = axis - center
        length = 2 * np.sqrt(np.sum(axis[0] * axis[0]))
        width = 2 * np.sqrt(np.sum(axis[1] * axis[1]))
        angle = np.arctan2(axis[0][1], axis[0][0])
        return center, (length, width), angle

    def project_points(self, points):
        return np.float32([[100 * x + 250, 100 * y + 250] for x, y in points])

    def draw_axis_camera(self, image, pnts):
        cv2.line(image, (pnts[0][0], pnts[0][1]), (pnts[1][0], pnts[1][1]), RED, 2)
        cv2.line(image, (pnts[0][0], pnts[0][1]), (pnts[2][0], pnts[2][1]), GREEN, 2)
        cv2.circle(image, (int(pnts[0][0]), int(pnts[0][1])), 3, BLUE, -1)
        return image

    def draw_TCP_axis(self, image):
        points = np.float32([[0, 0], [1, 0], [0, 1]])
        points = self.project_points(points)
        #pnt_TCP_final = self.transform(self.hom_vis, pxls_TCP)
        image = self.draw_axis_camera(image, points)
        return image

    def draw_points(self, image, pnts):
        pxls = self.transform(self.hom, pnts)
        pnts_final = self.transform(self.hom_vis, pxls)
        for pnt in pnts_final:
            cv2.circle(image, (int(pnt[0]), int(pnt[1])), 3, BLUE, -1)
        return image

    def draw_ellipse(self, image, ellipse, color=(0, 0, 255)):
        center, axis, angle = ellipse
        center = self.project_points(np.float32([center]))[0]
        center = (int(center[1]), int(center[0]))
        axis = (int(round(100*axis[1]/2)), int(round(100*axis[0]/2)))
        angle = np.rad2deg(angle)
        cv2.ellipse(image, center, axis, angle, 0, 360, color, 2)
        cv2.circle(image, center, 3, color, -1)
        return image

    def draw_arrow(self, image, speed, vel, color=(0, 255, 255)):
        if speed > 0:
            print speed, vel
            if speed <= 10:
                scale = 0.1
            else:
                scale = 0.01
            points = np.float32([[0, 0], [scale * vel[0], scale * vel[1]]])
            points = self.project_points(points)
            p = (int(points[0][1]), int(points[0][0]))
            q = (int(points[1][1]), int(points[1][0]))
            arrow_magnitude = 10 * scale * speed
            angle = np.arctan2(p[1]-q[1], p[0]-q[0])
            p1 = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/6)),
                  int(q[1] + arrow_magnitude * np.sin(angle + np.pi/6)))
            p2 = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/6)),
                  int(q[1] + arrow_magnitude * np.sin(angle - np.pi/6)))
            cv2.line(image, p, q, color, 2)
            cv2.line(image, p1, q, color, 2)
            cv2.line(image, p2, q, color, 2)
        return image


if __name__ == '__main__':
    p_uEye = Projection()
    p_uEye.load_configuration('../../config/camera.yaml')
    images_uEye = []
    files_uEye = glob.glob("../../data/calibration/vis/frame*.jpg")[0:2]
    for f in sorted(files_uEye):
        im_uEye = cv2.imread(f)
        im_uEye = p_uEye.project_image(im_uEye)
        im_uEye = p_uEye.draw_TCP_axis(im_uEye)
        images_uEye.append(im_uEye)
        cv2.imshow("Image: ", im_uEye)
        cv2.waitKey(0)

    p_NIT = Projection()
    p_NIT.load_configuration('../../config/tachyon.yaml')
    images_NIT = []
    files_NIT = glob.glob("../../data/calibration/nit/frame*.jpg")[0:2]
    for f in sorted(files_NIT):
        im_NIT = cv2.imread(f)
        im_NIT = p_NIT.project_image(im_NIT)
        im_NIT = p_NIT.draw_TCP_axis(im_NIT)
        images_NIT.append(im_NIT)
        cv2.imshow("Image: ", im_NIT)
        cv2.waitKey(0)

    for im_NIT, im_uEye in zip(images_NIT, images_uEye):
        im_final = cv2.addWeighted(im_uEye, 0.4, im_NIT, 0.6, 0)
        cv2.imshow("Image final", im_final)
        cv2.waitKey(0)

    im_final = p_NIT.draw_ellipse(im_final, ((0, 0), (5, 1.25), np.pi/4))
    im_final = p_NIT.draw_arrow(im_final, 10, (10, 5, 0))
    cv2.imshow("Image final", im_final)
    cv2.waitKey(0)

    center, axis, angle = (12, 12), (7, 3), 0
    center, axis, angle = p_NIT.transform_ellipse(center, axis, angle)
    im_final = p_NIT.draw_ellipse(im_final, (center, axis, angle))
    cv2.imshow("Image final", im_final)
    cv2.waitKey(0)
