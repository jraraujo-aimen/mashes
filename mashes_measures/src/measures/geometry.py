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

    def find_ellipse(self, img_bin):
        cnt = self.find_contour(img_bin)
        axis, angle, center = (0, 0), 0, (0, 0)
        if cnt is not None:
            if len(cnt) > 4:
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (h, v), angle = ellipse
                center = (x, y)
                if h >= v:
                    axis = (h, v)
                    angle = np.deg2rad(angle)
                else:
                    axis = (v, h)
                    angle = np.deg2rad(angle-90)
        return center, axis, angle

    def find_geometry(self, frame):
        img_bin = self.binarize(frame)
        center, axis, angle = self.find_ellipse(img_bin)
        return center, axis, angle

    def draw_geometry(self, frame, ellipse):
        center, axis, angle = ellipse
        center = (int(round(center[0])), int(round(center[1])))
        axis = (int(round(axis[0]/2)), int(round(axis[1]/2)))
        cv2.ellipse(
            frame, center, axis, np.rad2deg(angle), 0, 360, (0, 0, 255), 1)
        return frame


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = cv2.imread('../../data/frame0000.jpg')
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    geometry = Geometry(127)
    ellipse = geometry.find_geometry(frame)
    print ellipse

    img = cv2.resize(img, (frame.shape[0] * 10, frame.shape[1] * 10))
    center = (ellipse[0][0] * 10, ellipse[0][1] * 10)
    axis = (ellipse[1][0] * 10, ellipse[1][1] * 10)
    angle = ellipse[2]

    frame1 = geometry.draw_geometry(img, (center, axis, angle))

    img = cv2.imread('../../data/frame0001.jpg')
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ellipse = geometry.find_geometry(frame)
    print ellipse

    frame2 = geometry.draw_geometry(img, ellipse)

    plt.figure()
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), interpolation='none')
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), interpolation='none')
    plt.show()
