#!/usr/bin/env python
from pylab import *
import cv2
import math
import numpy as np


class Geometry():
    def __init__(self,value=128):
        self.threshold = value

    def greyscale(self, frame):
        """RGB to gray scale."""
        img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('ImageWindow_gray', img_grey)
        return img_grey

    def binarize(self, frame):
        """Image binarization."""
        #_, img_bin = cv2.threshold(frame, self.threshold, 255,
        #                            cv2.THRESH_BINARY)
        img_bin=frame.copy()
        img_bin[img_bin<=self.threshold]=0
        img_bin[img_bin>self.threshold]=255
        img_bin=img_bin.astype("uint8")
        return img_bin

    def find_contour(self, frame):
        """Find the main countour"""
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        cnt_len = len(contours)
        if cnt_len > 0:
            #find max area
            areas = [cv2.contourArea(cnt) for cnt in contours]
            max_area = max(areas)
            if (max_area > 5):
                    index_area = areas.index(max_area)
                    cnt = contours[index_area]
                    if(len(cnt)>4): # Para dibujar la elipse necesitamos 5 o mas lineas de contorno.
                        return cnt
                    return None
            return None
        else:
            return None

    def find_ellipse(self, contour):
        """"Find ellipse as a contour"""
        ellipse = cv2.fitEllipse(contour)
        return ellipse


def find_geometry(frame,value=128):
    melt_pool = Geometry(value)
    # Pre-processing
    img_grey = melt_pool.greyscale(frame)
    img_bin = melt_pool.binarize(img_grey)

    cnt = melt_pool.find_contour(img_bin)
    #print cnt
    if cnt is not None:
        ellipse = melt_pool.find_ellipse(cnt)
        (x, y), (h, v), angle = ellipse
        angle_rads = math.radians(angle)
        major_axis = max(h, v)
        minor_axis = min(h, v)
        return (major_axis, minor_axis, angle_rads,x,y)
    return (None, None, None, None, None)


if __name__ == '__main__':
    img = cv2.imread('../../data/frameErr.png')
    img=(img.astype("uint16"))*(65535/255)
    (major_axis, minor_axis, angle_rads,x,y) = find_geometry(img,32000)
    print major_axis, minor_axis, angle_rads
    print "finish"
