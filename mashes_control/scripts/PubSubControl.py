#!/usr/bin/env python
import rospy
#from std_msgs.msg import String
from mashes_control.msg import MsgMode
from mashes_control.msg import MsgControl
from mashes_control.msg import MsgPower
from mashes_labjack.msg import MsgLabJack
from mashes_measures.msg import MsgGeometry

from control.control import Control
from control.control import PID


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

        self.mode = 0
        self.set_point = 0

        self.control = Control()
        self.PID = PID()
        # self.PID.setKp(kp)
        # self.PID.setKd(kd)
        # self.PID.setKi(ki)

        rospy.spin()

    def cb_mode(self, msg_mode):
        self.mode = msg_mode.value
        rospy.loginfo('Mode: ' + str(self.mode))

    def cb_control(self, msg_control):
        self.set_point = msg_control.set_point
        rospy.loginfo('Set Point: ' + str(self.set_point))

    def cb_geometry(self, msg_geo):
        if self.mode:
            self.msg_power.value = self.set_point
        else:
            minor_axis = msg_geo.minor_axis
            major_axis = msg_geo.major_axis
            value = self.PID.update(minor_axis * 214.28)
            #power_out = self.control.auto_output(self.power, self.set_point)
            #self.msg_labjack.value = power_out
            print 'Power:', value
            self.msg_power.value = value
        self.pub_power.publish(self.msg_power)


if __name__ == '__main__':
    try:
        PubSubControl()
    except rospy.ROSInterruptException:
        pass
