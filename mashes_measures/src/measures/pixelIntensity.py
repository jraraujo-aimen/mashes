#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @file        pixelIntensity.py
# @author       jacobo Costas
# @version     1.0
# @date        Created:     14/3/2016
#              Last Update: 
# ------------------------------------------------------------------------
# Description: Incluye clase para calcular la elipse utilizando momentos 
#              Utiliza imagen en escala de grises con pesos
# ------------------------------------------------------------------------
# History:     1.0 Se Cambia el estilo y se crea la clase
# ------------------------------------------------------------------------

import time
from pylab import *
import numpy as np
from matplotlib.patches import Rectangle, Ellipse
import cv2



class PixelIintensity():
        
    def __init__(self, max=None, min=None, value=255):
        ''' COnstructor de clase utiliando parametros
        '''
        self.max=200
        self.min=128
        if (max is not None) or (max>value):
            self.max=max
        if (min is not None) or (min>value):
            self.min=min  
            
            
        self.lut_conv = (np.asarray(range(value))*(256))/value

        #print self.lut_conv
        
        
        
    def binarize(self, frame,threshold):
        """Image binarization."""
        #_, img_bin = cv2.threshold(frame, self.threshold, 255,
        #                            cv2.THRESH_BINARY)
        img_bin=frame.copy()
        img_bin[img_bin<=threshold]=0
        img_bin[img_bin>threshold]=255
        img_bin=img_bin.astype("uint8")
        return img_bin
    
    
    def measure(self,image):
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
    
    
    def moments(self,image):
         """Calculates the moments of the shape in the image."""
         width, height = image.shape[1], image.shape[0]
         m00, m01, m10, m11, m02, m20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
         for y in range(height):
             for x in range(width):
                 #if image[y,x]:
                     m00 = m00 + image[y,x]
                     m10 = m10 + x*image[y,x]
                     m01 = m01 + y*image[y,x]
                     m11 = m11 + x * y*image[y,x]
                     m20 = m20 + x * x*image[y,x]
                     m02 = m02 + y * y*image[y,x]
 
                    
         return [m00, m01, m10, m11, m02, m20]
    
    def ellipse(self,image):
        """Calculates the ellipse approximation of the shape in the image."""
        m00, m01, m10, m11, m02, m20 = self.moments(image)
        x, y = mgrid[:image.shape[0],:image.shape[1]]

        x_cm, y_cm, angle, length, width = 0, 0, 0, 0, 0
        if m00 > 1000:
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
            length = 4 * sqrt(lmax)
            width = 4 * sqrt(lmin)
        return [x_cm, y_cm, angle, length, width]
    
    
   
    def pixel_intensity(self,image):
        """pixel intensity calculation"""
        img = image.copy()
        
        img[img>self.max] = self.max
        img[img<self.min] = 0
        
        img=self.lut_conv[img]
        #print img
        #print image
        
        return img

if __name__ == '__main__':
    import cv2

    img = cv2.imread('../../data/frame0015.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=(img.astype("uint16"))*(1024/255)
    print np.max(img)
    
    pixelInt = PixelIintensity(750,500,1024)

    bin = pixelInt.binarize(img, 750)
    x_cm, y_cm, angle, length, width = pixelInt.ellipse(bin)
    print angle, length, width
    
    pixel= pixelInt.pixel_intensity(img)
    x_cm_pixel, y_cm_pixel, angle_pixel, length_pixel, width_pixel = pixelInt.ellipse(pixel)
    print angle_pixel, length_pixel, width_pixel
     
     
    left, top, right, bottom = pixelInt.measure(img)
    print right - left, bottom - top

    figure()
    subplot(321)
    imshow(bin, cmap='gray')
    axis('off')
    subplot(322)
    imshow(pixel, cmap='jet')
     
    axis('off')
    subplot(323)
    imshow(img, cmap='gray')
    axis('off')
    subplot(324)
    imshow(img, cmap='gray')
    axis('off')
    gca().add_patch(Rectangle([left, top], right - left, bottom - top, facecolor='none', edgecolor='green', lw=1.5))
    gca().add_patch(Ellipse([x_cm, y_cm], length, width, angle=rad2deg(angle), facecolor='none', edgecolor='red', lw=2.5))
     
    subplot(326)
    imshow(img, cmap='gray')
    axis('off')
    gca().add_patch(Rectangle([left, top], right - left, bottom - top, facecolor='none', edgecolor='green', lw=1.5))
    gca().add_patch(Ellipse([x_cm_pixel, y_cm_pixel], length_pixel, width_pixel, angle=rad2deg(angle_pixel), facecolor='none', edgecolor='red', lw=2.5))
 
    show()

