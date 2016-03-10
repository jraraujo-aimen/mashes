#!/usr/bin/env python
import rospy
from mashes_labjack.msg import MsgLabJack

from labjack.labjack import LabJack
from laser.laser import Laser


class SubLabjack():
    def __init__(self):
        self.dacs = LabJack()
        self.las = Laser()

    def callback(self, msg_labjack):
        rospy.loginfo(rospy.get_caller_id() + "Value %.2f", msg_labjack.value)
        output = msg_labjack.value * self.las.factor
        rospy.loginfo(output)
        self.dacs.output(output)

    def run(self):
        rospy.init_node('sublabjack')
        rospy.Subscriber("/control/out", MsgLabJack, self.callback)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()


if __name__ == '__main__':
    subla = SubLabjack()
    subla.run()
