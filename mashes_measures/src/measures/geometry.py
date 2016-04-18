import cv2
import numpy as np


class Geometry():
    def __init__(self, threshold=127):
        self.threshold = threshold

    def binarize(self, frame):
        _, img_bin = cv2.threshold(
            frame, self.threshold, 255, cv2.THRESH_BINARY)
        return img_bin

    def find_contour(self, frame):
        """Find the main countour"""
        contours, hierarchy = cv2.findContours(
            frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_len = len(contours)
        if cnt_len > 0:
            areas = [cv2.contourArea(cnt) for cnt in contours]
            max_area = max(areas)
            if (max_area > 5):
                index_area = areas.index(max_area)
                cnt = contours[index_area]
                return cnt
            return None
        else:
            return None

    def find_ellipse(self, contour):
        ellipse = cv2.fitEllipse(contour)
        return ellipse

    def find_geometry(self, frame):
        img_bin = self.binarize(frame)
        cnt = self.find_contour(img_bin)
        if cnt is not None:
            ellipse = self.find_ellipse(cnt)
            (x, y), (h, v), angle = ellipse
            angle_rads = np.deg2rad(angle)
            major_axis = max(h, v)
            minor_axis = min(h, v)
        else:
            major_axis, minor_axis, angle_rads = 0, 0, 0
        return major_axis, minor_axis, angle_rads


if __name__ == '__main__':
    geometry = Geometry()

    img = cv2.imread('../../data/frame0000.jpg')
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (major_axis, minor_axis, angle_rads) = geometry.find_geometry(frame)
    print major_axis, minor_axis, angle_rads

    # cv2.ellipse(img, ellipse, (0, 0, 255), 2)
    # cv2.imshow('ImageWindow_bin_2', img)
    # cv2.waitKey()

    img = cv2.imread('../../data/frame0001.jpg')
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (major_axis, minor_axis, angle_rads) = geometry.find_geometry(frame)
    print major_axis, minor_axis, angle_rads

    img = cv2.imread('../../data/frame0000.png')
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (major_axis, minor_axis, angle_rads) = geometry.find_geometry(frame)
    print major_axis, minor_axis, angle_rads
