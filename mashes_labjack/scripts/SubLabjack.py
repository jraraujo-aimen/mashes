#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from mashes_labjack.msg import MsgLabJack

from labjack.labjack import LabJack


def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Value %.2f", data.value)
    dacs.output(data.value)


def sublabjack():
    rospy.init_node('sublabjack')

    rospy.Subscriber("/labjack/value", MsgLabJack, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    dacs = LabJack()
    sublabjack()
