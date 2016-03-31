#!/usr/bin/env python
import os
import rospy
import rospkg

#from std_msgs.msg import String
from mashes_control.msg import MsgMode
from mashes_control.msg import MsgControl
from mashes_control.msg import MsgPower
from mashes_labjack.msg import MsgLabJack
from mashes_measures.msg import MsgGeometry

from control.control import Control
from control.control import PID


MANUAL = 0
AUTOMATIC = 1
path = rospkg.RosPack().get_path('mashes_control')


class PubSubControl():
    def __init__(self):
        rospy.init_node('pub_sub_control')

        rospy.Subscriber(
            '/tachyon/geometry', MsgGeometry, self.cb_geometry, queue_size=1)
        rospy.Subscriber(
            '/control/mode', MsgMode, self.cb_mode, queue_size=1)
        rospy.Subscriber(
            '/control/parameters', MsgControl, self.cb_control, queue_size=1)

        self.pub_power = rospy.Publisher(
            '/control/power', MsgPower, queue_size=10)

        self.msg_power = MsgPower()
        self.msg_labjack = MsgLabJack()

        self.mode = MANUAL
        self.set_point = 0

        self.control = Control()
        self.control.load_conf(os.path.join(path, 'config/control.yaml'))
        self.control.pid.setPoint(self.set_point)

        rospy.spin()

    def cb_mode(self, msg_mode):
        self.mode = msg_mode.value
        rospy.loginfo('Mode: ' + str(self.mode))

    def cb_control(self, msg_control):
        self.set_point = msg_control.set_point
        # Kp = msg_control.kp
        # Ki = msg_control.ki
        # Kd = msg_control.kd
        self.control.pid.setPoint(self.set_point)
        # self.control.pid.setParameters(Kp, Ki, Kd)
        rospy.loginfo('Set Point: ' + str(self.set_point))

    def cb_geometry(self, msg_geo):
        if self.mode == MANUAL:
            self.msg_power.value = self.set_point
        elif self.mode == AUTOMATIC:
            minor_axis = msg_geo.minor_axis
            value = self.control.pid.update(minor_axis)
            print 'Power (minor_axis):', value
            self.msg_power.value = value
        else:
            major_axis = msg_geo.major_axis
            value = self.control.pid.update(major_axis)
            print 'Power (major axis):', value
        self.pub_power.publish(self.msg_power)


if __name__ == '__main__':
    try:
        PubSubControl()
    except rospy.ROSInterruptException:
        pass
