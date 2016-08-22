import cv2
import numpy as np

from homography import Homography

BLUE = [255, 0, 0]
GREEN = [0, 255, 0]
RED = [0, 0, 255]


class Projection():
    def __init__(self, config=None):
        self.hom = Homography()
        self.vis = Homography()
        self.hom_vis = np.eye(3)
        if config is not None:
            self.load_configuration(config)

    def load_configuration(self, filename):
        hom = self.hom.load(filename)
        points = np.float32([[-2.5, -2.5], [2.5, -2.5], [-2.5, 2.5], [2.5, 2.5]])
        pixels = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])
        self.vis.calculate(points, pixels)
        self.hom_vis = np.dot(self.vis.inv, hom)

    def transform_ellipse(self, center, axis, angle):
        major_u = axis[0] / 2 * np.cos(angle) + center[0]
        major_v = axis[0] / 2 * np.sin(angle) + center[1]
        minor_u = axis[1] / 2 * np.cos(angle) + center[0]
        minor_v = axis[1] / 2 * np.sin(angle) + center[1]
        center = self.hom.transform([center])[0]
        axis = self.hom.transform(np.float32([[major_u, major_v],
                                              [minor_u, minor_v]]))
        axis = axis - center
        length = 2 * np.sqrt(np.sum(axis[0] * axis[0]))
        width = 2 * np.sqrt(np.sum(axis[1] * axis[1]))
        angle = np.arctan2(axis[0][1], axis[0][0])
        return center, (length, width), angle

    def project_image(self, image):
        im_measures = cv2.warpPerspective(image, self.hom_vis, (500, 500))
        return im_measures


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    p_NIT = Projection()
    p_NIT.load_configuration('../../config/tachyon.yaml')

    plt.figure()
    plt.subplot(121)
    im_NIT = cv2.imread('../../data/nit_focus.jpg')
    plt.imshow(cv2.cvtColor(im_NIT, cv2.COLOR_BGR2RGB), interpolation='none')
    plt.subplot(122)
    im_NIT = p_NIT.project_image(im_NIT)
    plt.imshow(cv2.cvtColor(im_NIT, cv2.COLOR_BGR2RGB), interpolation='none')
    plt.show()
