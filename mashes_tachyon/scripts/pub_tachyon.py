#!/usr/bin/env python
import os
import cv2
import time
import rospy
import rospkg

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from tachyon.tachyon import Tachyon
from tachyon.nitdat import LUT_IRON


def tachyon():
    rospy.init_node('tachyon')

    image_topic = rospy.get_param('~image', '/tachyon/image')
    image_pub = rospy.Publisher(image_topic, Image, queue_size=5)

    bridge = CvBridge()

    mode = rospy.get_param('~mode', 'mono8')
    config_file = rospy.get_param('~config', 'tachyon.yml')

    path = rospkg.RosPack().get_path('mashes_tachyon')
    config_filename = os.path.join(path, 'config', config_file)

    tachyon = Tachyon(config=config_filename)
    tachyon.connect()
    
    tachyon.calibrate(24)
    
    while not rospy.is_shutdown():
        try:
            frame, header = tachyon.read_frame()
            if mode == 'rgb8':
                frame = LUT_IRON[frame]
            else:
                frame = np.uint8(frame >> 2)
            image_msg = bridge.cv2_to_imgmsg(frame, encoding=mode)
            image_msg.header.stamp = rospy.Time.now()
            image_pub.publish(image_msg)
        except CvBridgeError, e:
            print e
            
    tachyon.disconnect()
    tachyon.close()


if __name__ == '__main__':
    tachyon()
