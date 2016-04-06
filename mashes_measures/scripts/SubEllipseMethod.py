#!/usr/bin/env python
import cv2
import math
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from measures.pixelIntensity import PixelIintensity
from measures.geometry import Geometry

import numpy as np
from matplotlib.patches import Rectangle, Ellipse


class SubEllipseMethod():

    def __init__(self):
         rospy.init_node('Sub_Cpmpare', anonymous=True)
         image_topic = rospy.get_param('~image', '/tachyon/image')

         rospy.Subscriber(image_topic, Image, self.callback, queue_size=1)
         self.melt_pool = Geometry()
         self.bridge = CvBridge()
         self.pixelInt = PixelIintensity()

         try:
             rospy.spin()
         except KeyboardInterrupt:
            print "Shutting down ROS Image feature detector module"


    def callback(self, ros_data):
        try:
             self.stamp = ros_data.header.stamp
             #print "Stamp ",self.stamp
             self.frame = self.bridge.imgmsg_to_cv2(ros_data,'rgb8')

             img_gray= cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)

             img_base_binary = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
             img_base_pixel=img_base_binary.copy()
             img_base_geometry=img_base_binary.copy()
             img_base_all=img_base_binary.copy()

             bin = self.pixelInt.binarize(img_gray, 128)
             x_cm, y_cm, angle, length, width = self.pixelInt.ellipse(bin)
             print "Binarize"
             print x_cm, y_cm, angle, length, width

             x_cm=nan_to_num(x_cm).astype("uint16")
             y_cm=nan_to_num(y_cm).astype("uint16")
             length=nan_to_num(length).astype("uint16")
             width=nan_to_num(width).astype("uint16")

             cv2.ellipse(img_base_binary,(x_cm,y_cm),(length/2,width/2),rad2deg(angle),0,360,(0,255,0),1)

             # PIXEL Intensity

             intensity=self.pixelInt.pixel_intensity(img_gray)
             print "Maximo img_gray %d"%img_gray.max()
             cv2.imshow("Intensidades",cv2.resize(intensity,(100,100)))
             cv2.imshow("Imagen binarizada",cv2.resize(bin,(100,100)))
             x_cm_pixel, y_cm_pixel, angle_pixel, length_pixel, width_pixel = self.pixelInt.ellipse(intensity)
             print "Valores en intensidad"
             print  x_cm_pixel, y_cm_pixel, angle_pixel, length_pixel, width_pixel
             x_cm_pixel=nan_to_num(x_cm_pixel).astype("uint16")
             y_cm_pixel=nan_to_num(y_cm_pixel).astype("uint16")
             length_pixel=nan_to_num(length_pixel).astype("uint16")
             width_pixel=nan_to_num(width_pixel).astype("uint16")
             cv2.ellipse(img_base_pixel,(x_cm_pixel,y_cm_pixel),(length_pixel/2,width_pixel/2),rad2deg(angle_pixel),0,360,(0,0,255),1)

            # Metodo Geometry

             melt_pool = Geometry()
             img_bin = melt_pool.binarize(img_gray)
             cnt = melt_pool.find_contour(img_bin)
             if cnt is not None:
                ellipse = melt_pool.find_ellipse(cnt)
                (x, y), (h, v), angle = ellipse
                angle_rads = np.deg2rad(angle)
                major_axis = max(h, v)
                minor_axis = min(h, v)
                print x , y, angle_rads, major_axis, minor_axis
                cv2.ellipse(img_base_geometry, ellipse, (255, 0, 0), 1)
                cv2.ellipse(img_base_all, ellipse, (255, 0, 0), 1)
                cv2.ellipse(img_base_all,(x_cm,y_cm),(length/2,width/2),rad2deg(angle),0,360,(0,255,0),1)
                cv2.ellipse(img_base_all,(x_cm_pixel,y_cm_pixel),(length_pixel/2,width_pixel/2),rad2deg(angle_pixel),0,360,(0,0,255),1)
                cv2.imshow('ALL', cv2.resize(img_base_all,(300,300)))


             cv2.imshow("Binarizacion :",cv2.resize(img_base_binary,(200,200)))
             #cv2.imshow('Geometry', cv2.resize(img_base_geometry,(100,100)))
             cv2.imshow("Pixel Intentity",cv2.resize(img_base_pixel,(200,200)))
             cv2.imshow('Geometry', cv2.resize(img_base_geometry,(200,200)))

            # Juntamos todo en una imagen
             cv2.waitKey(3)
        except CvBridgeError, e:
            print e


if __name__ == '__main__':
    SubEllipseMethod()
