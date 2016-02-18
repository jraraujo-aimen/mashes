#!/usr/bin/env python
from pylab import *
import cv2


class Geometry():
    def __init__(self):
        self.threshold = 178

    def greyscale(self, frame):
        """RGB to gray scale."""
        img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('ImageWindow_gray', img_grey)
        return img_grey

    def binarize(self, frame):
        """Image binarization."""
        _, img_bin = cv2.threshold(frame, self.threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ImageWindow_bin', img_bin)
        return img_bin

    def find_contour(self, frame):
        """Find the main countour"""
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        #find max area
        areas = [cv2.contourArea(cnt) for cnt in contours]
        max_area = max(areas)
        if (max_area > 5):
                index_area = areas.index(max_area)
                cnt = contours[index_area]
                return cnt
        return None

    def find_ellipse(self, contour):
        """"Find ellipse as a contour"""
        ellipse = cv2.fitEllipse(contour)
        return ellipse


def find_geometry(frame):
    clad = Geometry()
    # Pre-processing
    img_grey = clad.greyscale(frame)
    img_bin = clad.binarize(img_grey)
    cnt = clad.find_contour(img_bin)
    if cnt is not None:
        ellipse = clad.find_ellipse(cnt)
        (x, y), (h, v), angle = ellipse
        angle_rads = math.radians(angle)
        major_axis = max(h, v)
        minor_axis = min(h, v)
        cv2.ellipse(img, ellipse, (0, 0, 255), 2)
        cv2.imshow('ImageWindow_bin_2', img)
        cv2.waitKey()
        return (major_axis, minor_axis, angle_rads)
    return None


if __name__ == '__main__':
    img = cv2.imread('../../../../../../files/13_bag/frame0465.jpg')
    (major_axis, minor_axis, angle_rads) = find_geometry(img)
    print major_axis, minor_axis, angle_rads

    img = cv2.imread('../../../../../../files/13_bag/frame0466.jpg')
    (major_axis, minor_axis, angle_rads) = find_geometry(img)
    print major_axis, minor_axis, angle_rads

    img = cv2.imread('../../../../../../files/13_bag/frame0467.jpg')
    (major_axis, minor_axis, angle_rads) = find_geometry(img)
    print major_axis, minor_axis, angle_rads

    img = cv2.imread('../../../../../../files/13_bag/frame0468.jpg')
    (major_axis, minor_axis, angle_rads) = find_geometry(img)
    print major_axis, minor_axis, angle_rads

    img = cv2.imread('../../../../../../files/13_bag/frame0469.jpg')
    (major_axis, minor_axis, angle_rads) = find_geometry(img)
    print major_axis, minor_axis, angle_rads

    img = cv2.imread('../../../../../../files/13_bag/frame0470.jpg')
    (major_axis, minor_axis, angle_rads) = find_geometry(img)
    print major_axis, minor_axis, angle_rads

    img = cv2.imread('../../../../../../files/13_bag/frame0471.jpg')
    (major_axis, minor_axis, angle_rads) = find_geometry(img)
    print major_axis, minor_axis, angle_rads

    img = cv2.imread('../../../../../../files/13_bag/frame0472.jpg')
    (major_axis, minor_axis, angle_rads) = find_geometry(img)
    print major_axis, minor_axis, angle_rads

    img = cv2.imread('../../../../../../files/13_bag/frame0473.jpg')
    (major_axis, minor_axis, angle_rads) = find_geometry(img)
    print major_axis, minor_axis, angle_rads

    img = cv2.imread('../../../../../../files/13_bag/frame0474.jpg')
    (major_axis, minor_axis, angle_rads) = find_geometry(img)
    print major_axis, minor_axis, angle_rads

    img = cv2.imread('../../../../../../files/13_bag/frame0475.jpg')
    (major_axis, minor_axis, angle_rads) = find_geometry(img)
    print major_axis, minor_axis, angle_rads

    print "finish"
