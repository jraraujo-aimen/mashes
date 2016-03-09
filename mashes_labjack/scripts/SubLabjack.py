#!/usr/bin/env python
import rospy
import yaml
import os
import rospy
import rospkg
from std_msgs.msg import String
from mashes_labjack.msg import MsgLabJack

from labjack.labjack import LabJack


class SubLabjack():
    def __init__(self):
        config_file = rospy.get_param('~config', 'labjack.yml')
        path = rospkg.RosPack().get_path('mashes_labjack')
        config_filename = os.path.join(path, 'config', config_file)
        with open(config_filename, "r") as ymlfile:
            cfg = yaml.load(ymlfile)
        self.potencia_max = float(cfg['configuration']['potencia_maxima'])
        self.potencia_min = float(cfg['configuration']['potencia_minima'])
        self.dacs = LabJack()

    def callback(self, msg_labjack):
        rospy.loginfo(rospy.get_caller_id() + "Value %.2f", msg_labjack.value)
        print (self.potencia_max - self.potencia_min)
        factor = 5.0 / (self.potencia_max - self.potencia_min)
        print (factor)
        output = msg_labjack.value * factor
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
