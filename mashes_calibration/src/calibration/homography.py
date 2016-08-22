import cv2
import numpy as np


class Homography():
    def __init__(self, pnts_pattern):
        self.pnts_pattern = pnts_pattern
        self.hom = np.eye(3)
        self.inv = np.eye(3)

    def calculate(self, pnts):
        self.hom, status = cv2.findHomography(pnts, self.pnts_pattern)
        self.inv = np.linalg.inv(self.hom)
        return self.hom

    def scale(self, hom):
        return np.mean([hom[0, 0], hom[1, 1]])

    def transform(self, hom, pixels):
        """Transforms pixels to points."""
        self.hom = hom
        pnts = np.float32([
            np.dot(self.hom, np.array([pnt[0], pnt[1], 1])) for pnt in pixels])
        pnts = np.float32([pnt / pnt[2] for pnt in pnts])
        return pnts[:, :2]

    def project(self, points):
        """Projects points as pixels."""
        pnts = np.float32([
            np.dot(self.inv, np.array([pnt[0], pnt[1], 1])) for pnt in points])
        pixels = np.float32([pnt / pnt[2] for pnt in pnts])
        return pixels[:, :2]

if __name__ == '__main__':
    pattern_points = np.float32([[0, 0], [6, 0], [0, 6], [6, 6]])
    img_points = np.float32([[8, 7], [25, 6], [9, 24], [25, 23]])

    h = Homography(pattern_points)
    hom = h.calculate(img_points)

    print hom, h.scale(hom)
    print np.around(h.transform(img_points), decimals=4)
    print np.around(h.project(pattern_points), decimals=4)
