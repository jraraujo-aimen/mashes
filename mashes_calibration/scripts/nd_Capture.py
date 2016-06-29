#!/usr/bin/env python
import os
import cv2
import rospy
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from mashes_measures.msg import MsgVelocity
from mashes_measures.msg import MsgGeometry

path = rospkg.RosPack().get_path('mashes_calibration')


class NdCapture():
    def __init__(self):
        rospy.init_node('capture')

        image_topic = rospy.get_param('~image', '/tachyon/image')
        rospy.Subscriber(image_topic, Image, self.cb_image, queue_size=1)
        self.bridge = CvBridge()

        rospy.Subscriber(
            '/tachyon/geometry', MsgGeometry, self.cb_geometry, queue_size=1)

        rospy.Subscriber(
            '/velocity', MsgVelocity, self.cb_velocity, queue_size=1)

        rospy.spin()


    def cb_image(self, data):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(data)
            if data.encoding == 'mono8':
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
        except CvBridgeError, e:
            print e





    def cb_mode(self, msg_mode):
        self.mode = msg_mode.value
        rospy.loginfo('Mode: ' + str(self.mode))

    def cb_control(self, msg_control):
        self.power = msg_control.power
        self.setpoint = msg_control.setpoint
        Kp = msg_control.kp
        Ki = msg_control.ki
        Kd = msg_control.kd
        self.control.pid.set_setpoint(self.setpoint)
        self.control.pid.set_parameters(Kp, Ki, Kd)
        rospy.loginfo('Params: ' + str(msg_control))

    def cb_geometry(self, msg_geo):
        stamp = msg_geo.header.stamp
        time = stamp.to_sec()
        if self.mode == MANUAL:
            value = self.control.pid.power(self.power)
        elif self.mode == AUTOMATIC:
            minor_axis = msg_geo.minor_axis
            if minor_axis > 0.5:
                value = self.control.pid.update(minor_axis, time)
            else:
                value = self.control.pid.power(self.power)
        else:
            major_axis = msg_geo.major_axis
            if major_axis:
                value = self.control.pid.update(major_axis, time)
            else:
                value = self.control.pid.power(self.power)
        self.msg_power.header.stamp = stamp
        self.msg_power.value = value
        print '# Timestamp', time, '# Power', self.msg_power.value
        self.pub_power.publish(self.msg_power)


if __name__ == '__main__':
    try:
        NdControl()
    except rospy.ROSInterruptException:
        pass





----------------------------------------------------------------








    def cb_shot(self, data):
        try:
            if self.frame_ok:
                if data.value == 1:
                    self.scale = ScaleCamera()
                    #prueba
                    pts_font = [[0, 0], [190, 0], [0, 190], [190, 190]]
                    print pts_font
                    pts_font = np.vstack(pts_font).astype(float)
                    print pts_font
                    im_src = cv2.imread('/home/noemi/catkin_ws/src/mashes/mashes_tf/scripts/pattern_trans_2.png')
                    #real
                    #self.capture_1 = self.frame
                    #print self.capture_1
                    #prueba: Read source image.
                    self.capture_1 = cv2.imread('/home/noemi/catkin_ws/src/mashes/mashes_tf/scripts/pattern_2.png')
                    print self.capture_1
                    self.cam_H_cal = self.scale.homography(im_src, pts_font, self.capture_1)

                    self.listen_tf()
                    self.base_H_tool = transformations.quaternion_matrix(self.rot)
                    print self.base_H_tool.size
                    print self.base_H_tool.shape
                    self.base_H_tool[3, 0] = self.trans[0]
                    self.base_H_tool[3, 1] = self.trans[1]
                    self.base_H_tool[3, 2] = self.trans[2]
                    print "base_H_tool"
                    print self.base_H_tool
                    print "\n"

                elif data.value == 2:
                    print "rotate"
                    #real
                    #self.capture_rotate = self.frame
                    #print self.capture_rotate
                    #prueba: Read source image.
                    self.capture_rotate = cv2.imread('/home/noemi/catkin_ws/src/mashes/mashes_tf/scripts/pattern_rotate.png')
                    self.scale.offset_TCP(self.capture_rotate)

                elif data.value == 3:
                    print "moved"
                    #self.capture_moved = self.frame
                    #print self.capture_moved
                    #prueba: Read source image.
                    self.capture_moved = cv2.imread('/home/noemi/catkin_ws/src/mashes/mashes_tf/scripts/pattern_moved.png')
                    self.tool_H_cam = self.scale.turn_TCP(self.capture_moved)

        except CvBridgeError, e:
            print e

    def cb_image(self, data):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(data)
            if data.encoding == 'mono8':
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
            if not self.frame_ok:
                self.frame_ok = True
        except CvBridgeError, e:
            print e

    def listen_tf(self):

        self.time = rospy.Time.now()
        self.listener.waitForTransform(
            "world", "tcp0", self.time, rospy.Duration(1.0))
        (trans, rot) = self.listener.lookupTransform(
            "world", "tcp0", self.time)
        self.trans = trans
        self.rot = rot

if __name__ == '__main__':
    try:
        nd_Capture()
    except rospy.ROSInterruptException:
        pass
