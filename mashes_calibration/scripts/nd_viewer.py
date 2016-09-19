#!/usr/bin/env python
import os
import tf
import cv2
import rospy
import rospkg
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from tachyon.tachyon import LUT_IRON

rospack = rospkg.RosPack()
path = rospack.get_path('mashes_calibration')


class ImageViewer():
    def __init__(self):
        rospy.init_node('viewer', anonymous=True)

        image_topic = rospy.get_param('~image', '/camera/image')
        self.camera = image_topic.split('/')[1]

        rospy.Subscriber(image_topic, Image, self.callback, queue_size=1)
        rospy.on_shutdown(self.on_shutdown_hook)

        self.counter = 0
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        cv2.namedWindow('viewer')
        cv2.cv.SetMouseCallback('viewer', self.on_mouse, '')

        rospy.spin()

    def on_shutdown_hook(self):
        cv2.destroyWindow('viewer')

    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.cv.CV_EVENT_RBUTTONDOWN:
            self.counter += 1
            filename = os.path.join(
                path, 'data', 'frame%04i.png' % self.counter)
            cv2.imwrite(filename, self.frame)
            rospy.loginfo(filename)
            try:
                self.listener.waitForTransform(
                    '/base_link', '/tool0', self.stamp, rospy.Duration(1.0))
                transform = self.listener.lookupTransform(
                    '/base_link', '/tool0', self.stamp)  # (trans, rot)
                filename = os.path.join(
                    path, 'data', 'pose%04i.txt' % self.counter)
                with open(filename, 'w') as f:
                    f.write(str(transform))
                rospy.loginfo(transform)
            except:
                rospy.loginfo('The transformation is not accesible.')

    def callback(self, data):
        try:
            self.stamp = data.header.stamp
            frame = self.bridge.imgmsg_to_cv2(data)
            if data.encoding == 'mono16':
                if self.camera == 'tachyon':
                    self.frame = cv2.cvtColor(LUT_IRON[frame], cv2.COLOR_RGB2BGR)
                else:
                    self.frame = frame
            elif data.encoding == 'mono8':
                self.frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                self.frame = frame
            self.show_frame()
        except CvBridgeError, e:
            print e

    def show_frame(self):
        h, w = self.frame.shape[:2]
        s = 128 / w
        if s > 0:
            frame = cv2.resize(self.frame, (s * w, s * h))
        else:
            frame = self.frame
        cv2.imshow("viewer", frame)
        cv2.waitKey(20)


if __name__ == '__main__':
    ImageViewer()
