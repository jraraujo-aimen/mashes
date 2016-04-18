#!/usr/bin/env python
import rospy
from mashes_labjack.msg import MsgLabJack

from labjack.labjack import LabJack
from laser.laser import Laser


class SubLabjack():
    def __init__(self):
        self.dacs = LabJack()
        self.las = Laser()
        self.laser_line_percent = rospy.get_param('/laser_line_percent')
        print self.laser_line_percent
        self.set_laser_power(self.laser_line_percent)

    def callback(self, msg_labjack):
        self.dacs.reg = 5000
        rospy.loginfo(rospy.get_caller_id() + "Value %.2f", msg_labjack.value)
        output = msg_labjack.value * self.las.factor
        rospy.loginfo(output)
        self.dacs.output(output)

    def set_laser_power(self, laser):
        output = 5 - 0.05*laser
        rospy.loginfo(output)
        self.dacs.reg = 5002
        self.dacs.output(output)




    def run(self):
        rospy.init_node('sublabjack')
        rospy.Subscriber("/control/out", MsgLabJack, self.callback)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()


if __name__ == '__main__':
    subla = SubLabjack()
    subla.run()
