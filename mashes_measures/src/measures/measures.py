#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @file        measures.py
# @author      Jorge Araujo, jacobo Costas
# @version     1.0
# @date        Created:     -
#              Last Update: 16/3/2016
# ------------------------------------------------------------------------
# Description: Incluye clase para calcular la elipse utilizando momentos
#              Utiliza el sistema de binarizar la imagen en estala de grises
# ------------------------------------------------------------------------
# History:
#              1.0     Inicial
# ------------------------------------------------------------------------


import time
from pylab import *
import numpy as np
from matplotlib.patches import Rectangle, Ellipse


class Measures():
    ''' Contiene las funciones para calculo de los momento utilziando una
        representacion de la imagen binarizada
    '''
    @staticmethod
    def binarize(image,threshold):
        """Binarizes the image through the configured thresholds."""
        width ,height = image.shape[1], image.shape[0]
        threshold = uint8(threshold)
        img = zeros((height, width), dtype=uint8)
        for y in range(height):
            for x in range(width):
                if image[y ,x] > threshold:
                    img[y ,x] = 255
        img=img.astype("uint8")
        return img

    @staticmethod
    def erosion(image):
        """Erodes the image with a 3x3 kernel."""
        width, height = image.shape[1], image.shape[0]
        img = zeros((height, width), dtype=uint8)
        for y in range(height - 2):
            for x in range(width - 2):
                if (image[y,x] and image[y,x+1] and image[y,x+2] and
                    image[y+1,x] and image[y+1,x+1] and image[y+1,x+2] and
                    image[y+2,x] and image[y+2,x+1] and image[y+2,x+2]):
                    img[y,x] = 255
        return img

    @staticmethod
    def dilation(image):
        """Dilates the image with a 3x3 kernel."""
        width, height = image.shape[1], image.shape[0]
        img = zeros((height, width), dtype=uint8)
        for y in range(height - 2):
            for x in range(width - 2):
                if (image[y,x] or image[y,x+1] or image[y,x+2] or
                    image[y+1,x] or image[y+1,x+1] or image[y+1,x+2] or
                    image[y+2,x] or image[y+2,x+1] or image[y+2,x+2]):
                    img[y,x] = 255
        return img

    @staticmethod
    def measure(image):
        """Locates the boundary box of the shape in the image."""
        size = 0
        width, height = image.shape[1], image.shape[0]
        # Boundary box
        left, top = width, height
        right, bottom = 0, 0
        for y in range(height):
            for x in range(width):
                if image[y, x]:
                    size = size + 1
                    if left > x:
                        left = x
                    if top > y:
                        top = y
                    if right < x:
                        right = x
                    if bottom < y:
                        bottom = y
        if size:
            box = [left, top, right, bottom]
        else:
            box = [0, 0, 0, 0]
        return box

    @staticmethod
    def moments(image):
        """Calculates the moments of the shape in the image."""
        width, height = image.shape[1], image.shape[0]
        m00, m01, m10, m11, m02, m20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for y in range(height):
            for x in range(width):
                if image[y,x]:
                    m00 = m00 + 1
                    m10 = m10 + x
                    m01 = m01 + y
                    m11 = m11 + x * y
                    m20 = m20 + x * x
                    m02 = m02 + y * y
        return [m00, m01, m10, m11, m02, m20]

    @staticmethod
    def ellipse(image):
        """Calculates the ellipse approximation of the shape in the image."""
        m00, m01, m10, m11, m02, m20 = Measures.moments(image)
        x_cm, y_cm, angle, length, width = 0, 0, 0, 0, 0
        if m00 > 10:
            x_cm = m10 / m00
            y_cm = m01 / m00

            #-- Rounded -------
            m11 = m11 / 64 * 64
            m20 = m20 / 64 * 64
            m02 = m02 / 64 * 64
            #------------------

            u20 = float(m20 / m00 - x_cm * x_cm)
            u02 = float(m02 / m00 - y_cm * y_cm)
            u11 = float(m11 / m00 - x_cm * y_cm)
            lmax = (u20 + u02 + sqrt(4 * u11 * u11 + (u20 - u02) * (u20 - u02))) / 2
            lmin = (u20 + u02 - sqrt(4 * u11 * u11 + (u20 - u02) * (u20 - u02))) / 2

            angle = np.arctan((2 * u11) / (u20 - u02)) / 2
            length = 4 * np.sqrt(lmax)
            width = 4 * np.sqrt(lmin)
        return [x_cm, y_cm, angle, length, width]

    @staticmethod
    def memory_bin(image, filename):
        """Generates the memory initialization file for the testbench simulation."""
        memory = ''
        width, height = image.shape[1], image.shape[0]
        for y in range(height):
            for x in range(width):
                binary = bin(256 | int(image[y,x]))[-8:]
                memory += binary + '\n'
        file = open(filename, 'w')
        file.write(memory)
        file.close()



if __name__ == '__main__':
    import cv2

    img = cv2.imread('../../data/frame0000.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bin = Measures.binarize(img, 128)

    x_cm, y_cm, angle, length, width = Measures.ellipse(bin)
    print angle, length, width

    left, top, right, bottom = Measures.measure(bin)
    print right - left, bottom - top

    figure()
    subplot(221)
    imshow(img, cmap='gray')
    axis('off')
    subplot(222)
    imshow(bin, cmap='gray')
    axis('off')
    subplot(223)
    imshow(bin, cmap='gray')
    axis('off')
    subplot(224)
    imshow(img, cmap='gray')
    axis('off')
    gca().add_patch(Rectangle([left, top], right - left, bottom - top, facecolor='none', edgecolor='green', lw=1.5))
    gca().add_patch(Ellipse([x_cm, y_cm], length, width, angle=rad2deg(angle), facecolor='none', edgecolor='red', lw=2.5))
    show()
