#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @file        PubPixelIntensity.py
# @author      Jacobo Costas
# @version     1.0
# @date        Created:     16/3/2016
#              Last Update: 
# ------------------------------------------------------------------------
# Description: Crea un nodo de ROS que recoge una imagen de un topic y 
#              calcula los momentos binarizando la imagen.
#              Publica los valores en otro topic     
# ------------------------------------------------------------------------
# History:     1.0 Inicial
# ------------------------------------------------------------------------
import rospy
import cv2
import math
from mashes_measures.msg import MsgMeasure
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from measures.measures import Measures
import numpy as np
import cv2


class PubMeasure():
    def __init__(self):
        rospy.init_node('pub_pixel_intensity', anonymous=True)

        image_topic = rospy.get_param('~image', '/tachyon/image')
        rospy.Subscriber(image_topic, Image, self.callback, queue_size=1)
        self.bridge = CvBridge()

        camera = image_topic.split('/')[1]
        pixel_topic = '/%s/Measure' % camera
        self.pixel_pub = rospy.Publisher(pixel_topic, MsgMeasure, queue_size=5)
        self.msg_pixel = MsgMeasure()
        #self.pixel_intensity = PixelIintensity()
        rospy.spin()

    def callback(self, data):
        try:
            self.stamp = data.header.stamp
            self.frame = self.bridge.imgmsg_to_cv2(data)

            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
            bin = Measures.binarize(self.frame,128)
            print bin.dtype
            x_cm, y_cm, angle, length, width = Measures.ellipse(bin)        

            angle_rads = 0
            major_axis = 0
            minor_axis = 0
            x          = 0
            y          = 0
            color = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2RGB)
            if (length >0) and (width >0):
                major_axis=(length)
                minor_axis=(width)
                angle_rads=angle
                x         =(x_cm)
                y         =(y_cm)
                
                x_16=np.uint16(x_cm)
                y_16=np.uint16(y_cm)
                angle_16=np.rad2deg(angle_rads).astype("uint16") 
                major_16=major_axis.astype("uint16")
                minor_16=minor_axis.astype("uint16")
                print"Measures : Angle: %s, half Major axis: %s, Half Minor axis: %s, x: %s, y: %s" % (angle_rads, major_axis, minor_axis, x ,y)
                cv2.ellipse(color,(x_16,y_16), (major_16/2,minor_16/2),angle_16,0,360,(0,255,0),1)
                    
            self.msg_pixel.header.stamp = self.stamp
            self.msg_pixel.orientation = angle_rads
            self.msg_pixel.major_axis = major_axis
            self.msg_pixel.minor_axis = minor_axis
            self.msg_pixel.x = x
            self.msg_pixel.y = y
            self.pixel_pub.publish(self.msg_pixel)
            
            cv2.imshow("Pixel_Measure",cv2.resize(color,(200,200)))
            cv2.waitKey(3)
        except CvBridgeError, e:
            print e


if __name__ == '__main__':
    try:
        PubMeasure()
    except rospy.ROSInterruptException:
        pass
