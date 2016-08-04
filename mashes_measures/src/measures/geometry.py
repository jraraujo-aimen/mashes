import cv2
import numpy as np


class Geometry():
    def __init__(self, threshold=127):
        self.threshold = threshold

    def binarize(self, frame):
        img_bin = np.zeros(frame.shape, dtype=np.uint8)
        img_bin[frame > self.threshold] = 255
        return img_bin

    def find_contour(self, frame):
        contours, hierarchy = cv2.findContours(
            frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = None
        if len(contours) > 0:
            areas = [cv2.contourArea(contour) for contour in contours]
            if np.max(areas) > 5:
                cnt = contours[np.argmax(areas)]
        return cnt

    def find_ellipse(self, contour):
        ellipse = cv2.fitEllipse(contour)
        return ellipse

    def find_geometry(self, frame):
        img_bin = self.binarize(frame)
        cnt = self.find_contour(img_bin)
        axis, angle, center = (0, 0), 0, (0, 0)
        if cnt is not None:
            if len(cnt) > 4:
                ellipse = self.find_ellipse(cnt)
                (x, y), (h, v), angle = ellipse
                center = (x, y)
                if h >= v:
                    angle = np.deg2rad(angle)
                    axis = (h, v)
                else:
                    angle = np.deg2rad(angle-90)
                    axis = (v, h)
        return center, axis, angle

    def draw_geometry(self, frame, ellipse):
        center, axis, angle = ellipse
        center = (int(round(center[0])), int(round(center[1])))
        axis = (int(round(axis[0]/2)), int(round(axis[1]/2)))
        cv2.ellipse(
            frame, center, axis, np.rad2deg(angle), 0, 360, (0, 0, 255), 1)
        return frame


if __name__ == '__main__':
    geometry = Geometry()

    img = cv2.imread('../../data/frame0000.jpg')
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ellipse = geometry.find_geometry(frame)
    print ellipse

    img = cv2.resize(img, (frame.shape[0] * 10, frame.shape[1] * 10))
    center = (ellipse[0][0] * 10, ellipse[0][1] * 10)
    axis = (ellipse[1][0] * 10, ellipse[1][1] * 10)
    angle = ellipse[2]

    frame = geometry.draw_geometry(img, (center, axis, angle))
    cv2.imshow('Image', frame)
    cv2.waitKey()

    img = cv2.imread('../../data/frame0001.jpg')
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ellipse = geometry.find_geometry(frame)
    print ellipse

    frame = geometry.draw_geometry(img, ellipse)
    cv2.imshow('Image', frame)
    cv2.waitKey()
