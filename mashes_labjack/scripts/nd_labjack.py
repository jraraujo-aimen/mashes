#!/usr/bin/env python
import os
import rospy
import rospkg

from mashes_control.msg import MsgPower

from labjack.labjack import LabJack


path = rospkg.RosPack().get_path('mashes_labjack')


class NdLabjack():
    def __init__(self):
        rospy.init_node('labjack')

        power_min = rospy.get_param('~power_min', 0)
        power_max = rospy.get_param('~power_max', 1500)

        self.labjack = LabJack()
        self.labjack.power_factor(power_min, power_max)
        self.laser_line_percent = rospy.get_param('/laser_line_percent')
        self.set_laserline_power(self.laser_line_percent)

        rospy.Subscriber("/control/power", MsgPower, self.cb_power)
        rospy.spin()

    def cb_power(self, msg_power):
        self.labjack.reg = 5000
        output = self.labjack.factor * msg_power.value
        rospy.loginfo("Power: %.2f, Output: %.2f", msg_power.value, output)
        self.labjack.output(output)

    def set_laserline_power(self, laser):
        output = 5 - 0.05*laser
        rospy.loginfo(output)
        self.labjack.reg = 5002
        self.labjack.output(output)


if __name__ == '__main__':
    try:
        NdLabjack()
    except rospy.ROSInterruptException:
        pass
