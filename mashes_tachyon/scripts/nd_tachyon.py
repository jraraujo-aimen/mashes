#!/usr/bin/env python
import os
import cv2
import rospy
import rospkg
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from mashes_tachyon.msg import MsgCalibrate
from mashes_tachyon.msg import MsgTemperature

from tachyon.tachyon import Tachyon
from tachyon.tachyon import LUT_IRON


class NdTachyon():
    def __init__(self):
        rospy.init_node('tachyon')

        image_topic = rospy.get_param('~image', '/tachyon/image')
        image_pub = rospy.Publisher(image_topic, Image, queue_size=10)
        self.pub_temp = rospy.Publisher(
            '/tachyon/temperature', MsgTemperature, queue_size=10)
        self.msg_temp = MsgTemperature()

        rospy.Subscriber(
            '/tachyon/calibrate', MsgCalibrate, self.cb_calibrate, queue_size=1)

        bridge = CvBridge()

        mode = rospy.get_param('~mode', 'mono16')
        config_file = rospy.get_param('~config', 'tachyon.yml')

        path = rospkg.RosPack().get_path('mashes_tachyon')
        config_filename = os.path.join(path, 'config', config_file)

        tachyon = Tachyon(config=config_filename)
        tachyon.connect()

        self.calibrate = True
        while not rospy.is_shutdown():
            try:
                if self.calibrate:
                    tachyon.calibrate(24)
                    self.calibrate = False
                stamp = rospy.Time.now()
                frame, header = tachyon.read_frame()
                header = tachyon.parse_header(header)
                frame = tachyon.process_frame(frame)
                if mode == 'rgb8':
                    frame = LUT_IRON[frame]
                elif mode == 'mono16':
                    frame = np.uint16(frame)
                    frame = cv2.resize(frame[11:27,6:22], (32, 32))
                    # frame = cv2.resize(frame, (32, 32))
                else:
                    frame = np.uint8(frame >> 2)
                image_msg = bridge.cv2_to_imgmsg(frame, encoding=mode)
                image_msg.header.stamp = stamp
                self.msg_temp.header.stamp = stamp
                self.msg_temp.temperature = header['Temperature'] / 10.0
                image_pub.publish(image_msg)
                self.pub_temp.publish(self.msg_temp)
            except CvBridgeError, e:
                rospy.loginfo("CvBridge Exception")

        tachyon.disconnect()
        tachyon.close()

    def cb_calibrate(self, msg_calibrate):
        if msg_calibrate.calibrate:
            self.calibrate = True


if __name__ == '__main__':
    try:
        NdTachyon()
    except rospy.ROSInterruptException:
        pass
