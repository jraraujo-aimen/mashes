#!/usr/bin/env python
import os
import rospy
import rospkg

from mashes_control.msg import MsgPower

from labjack.labjack import LabJack


path = rospkg.RosPack().get_path('mashes_labjack')


class SubLabjack():
    def __init__(self):
        rospy.init_node('sub_labjack')

        self.labjack = LabJack()
        config_file = rospy.get_param('~config', 'labjack.yml')
        self.labjack.load_config(os.path.join(path, 'config', config_file))

        rospy.Subscriber("/control/power", MsgPower, self.cb_power)
        rospy.spin()

    def cb_power(self, msg_power):
        output = self.labjack.factor * msg_power.value
        rospy.loginfo("Power: %.2f, Output: %.2f", msg_power.value, output)
        self.labjack.output(output)


if __name__ == '__main__':
    try:
        SubLabjack()
    except rospy.ROSInterruptException:
        pass
