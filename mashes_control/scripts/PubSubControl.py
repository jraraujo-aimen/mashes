#!/usr/bin/env python
import rospy
#from std_msgs.msg import String
from mashes_control.msg import MsgModo
from mashes_labjack.msg import MsgLabJack
from mashes_control.msg import MsgPower
from mashes_control.msg import MsgControl
from mashes_control.msg import MsgPower_detected
from mashes_measures.msg import MsgGeometry
#from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from control.control import Control
from control.control import PID


class PubSubControl():
    def __init__(self):
        rospy.init_node('control')
        self.bridge = CvBridge()
        self.msg_modo = MsgModo()
        self.msg_power_detected = MsgPower_detected()
        self.msg_labjack = MsgLabJack()
        rospy.Subscriber(
            '/control/power', MsgPower, self.manual, queue_size=1)
        rospy.Subscriber(
            '/tachyon/geometry', MsgGeometry, self.automatico, queue_size=1)
        rospy.Subscriber('/control/mode', MsgModo, self.callback, queue_size=1)
        rospy.Subscriber('/control/parameters', MsgControl,
                         self.parameters_control, queue_size=1)
        self.pub = rospy.Publisher(
            '/control/out', MsgLabJack, queue_size=10)
        # self.pub_po = rospy.Publisher(
        #     '/control/power_detected', MsgPower_detected, queue_size=10)
        self.mode = True
        self.control = Control()
        self.PID = PID()
        self.PID.setPoint(5.0)
        self.set_point = 500.0
        self.power = 0.0
        self.last_volt_value = 0.0
        self.minor_axis_max = 0.0
        self.minor_axis_min = 0.0

    def callback(self, msg_modo):
        self.mode = msg_modo.value
        rospy.loginfo(self.mode)
        if self.mode is True:
            rospy.loginfo('Manual mode selected')
        if self.mode is False:
            rospy.loginfo('Automatic mode selected')

    def manual(self, msg_power):
        self.set_point = msg_power.value
        self.PID.setPoint(self.set_point)
        rospy.loginfo('New set point stablished: ' + str(self.set_point))
        if self.mode is True:
            self.msg_labjack.value = self.set_point
            self.pub.publish(self.msg_labjack)

    def automatico(self, msg_geo):
        minor_axis = msg_geo.minor_axis
        major_axis = msg_geo.major_axis
        print minor_axis
        valor = self.PID.update(minor_axis * 214.28)
        print ('Valor ' + str(valor))

        if self.mode is False:
            try:
                pass
                # cv_image = self.bridge.imgmsg_to_cv2(geometry_msg, "bgr8")
                # self.power = self.control.power_detection(cv_image)
                # rospy.loginfo('Current power: ' + str(self.power))
            #     power_out = self.control.auto_output(self.power,
            #                                          self.set_point)
            #     print (power_out)
            except CvBridgeError as e:
                rospy.logerr(e)
            #self.msg_labjack.value = power_out
            self.pub.publish(self.msg_labjack)
            # self.msg_power_detected.value = self.power
            #self.pub_po.publish(self.msg_power_detected)
            #rospy.loginfo('Power value: ' + str(power_out))

    def parameters_control(self, msg_control):
        kp = msg_control.kp
        self.PID.setKp(kp)
        kd = msg_control.kd
        self.PID.setKd(kd)
        ki = msg_control.ki
        self.PID.setKi(ki)


    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':

        control = PubSubControl()
        control.run()
