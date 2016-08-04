#!/usr/bin/env python
import os
import rospy
import rospkg

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from mashes_tachyon.msg import MsgCalibrate

import numpy as np
from tachyon.tachyon import Tachyon, LUT_IRON


class NdTachyon():
    def __init__(self):
        rospy.init_node('tachyon')

        image_topic = rospy.get_param('~image', '/tachyon/image')

        image_pub = rospy.Publisher(image_topic, Image, queue_size=10)

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
                frame, header = tachyon.read_frame()
                frame = tachyon.process_frame(frame)
                if mode == 'rgb8':
                    frame = LUT_IRON[frame]
                elif mode == 'mono16':
                    frame = np.uint16(frame)
                else:
                    frame = np.uint8(frame >> 2)
                image_msg = bridge.cv2_to_imgmsg(frame, encoding=mode)
                image_msg.header.stamp = rospy.Time.now()
                image_pub.publish(image_msg)
            except CvBridgeError, e:
                print e

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
