#!/usr/bin/env python

import rospy
import sys
import os
import rospkg
from mashes_control.msg import MsgModo
from mashes_control.msg import MsgPower
from mashes_control.msg import MsgPower_detected
from mashes_control.msg import MsgControl
from PyQt4 import QtCore, QtGui, uic
# Cargar nuestro archivo .ui
path = rospkg.RosPack().get_path('mashes_control')
form_class = uic.loadUiType(os.path.join(path, 'resources', 'gui_celd.ui'))[0]


class QControl(QtGui.QWidget, form_class):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.label_Status.setStyleSheet(
            "background-color: rgb(255, 0, 0); color: rgb(255, 255, 255);")
        self.label_Status.setText("Manual")
        self.mode = True
        self.btn_Mode.setText("Auto")
        self.label_Error.setText("")
        self.label_Error.setWordWrap(True)
        self.label_Power_detected.setWordWrap(True)

        self.label_Status.setAlignment(QtCore.Qt.AlignCenter)
        self.btn_Mode.clicked.connect(self.btn_Mode_clicked)
        self.btnControl.clicked.connect(self.btnclickedControl)
        self.edit_Power.returnPressed.connect(self.edit_Power_enter)
        self.PubSubFunctions()

    def PubSubFunctions(self):
        rospy.init_node('GUI', anonymous=True)
        mode_topic = '/control/mode'
        power_topic = '/control/power'
        power_detected_topic = '/control/power_detected'

        self.pub_modo = rospy.Publisher(mode_topic, MsgModo, queue_size=10)
        self.msg_modo = MsgModo()
        self.pub_power = rospy.Publisher(power_topic, MsgPower, queue_size=10)
        self.msg_power = MsgPower()
        self.pub_control = rospy.Publisher('control/parameters', MsgControl,
                                           queue_size=10)
        self.msg_control = MsgControl()
        rospy.Subscriber(power_detected_topic,
                         MsgPower_detected, self.callback_Power, queue_size=10)

 # Event from Power_detected topic
    def callback_Power(self, data):
        self.label_Power_detected.setText("Power_detected: " + str(data.value))

 # Event from btn_Mode button
    def btn_Mode_clicked(self):

        if self.mode:
            self.label_Status.setText("Auto")
            self.mode = False
            self.btn_Mode.setText("Manual")
        else:
            self.label_Status.setText("Manual")
            self.mode = True
            self.btn_Mode.setText("Auto")

        self.msg_modo.value = self.mode
        rospy.loginfo(self.mode)
        self.pub_modo.publish(self.msg_modo)

    def btnclickedControl(self):
        self.kp = float(self.Kp_edit.text())
        self.kd = float(self.Kd_edit.text())
        self.ki = float(self.Ki_edit.text())
        self.msg_control.kp = self.kp
        self.msg_control.kd = self.kd
        self.msg_control.ki = self.ki
        self.pub_control.publish(self.msg_control)




 # Event from btn_Power button
    def edit_Power_enter(self):
        power = float(self.edit_Power.text())
        if power < 1500:
            self.label_Error.setText("")
            self.edit_Power.clearFocus()
            self.msg_power.value = power
            rospy.loginfo(power)
            self.pub_power.publish(self.msg_power)

        else:
            self.label_Error.setStyleSheet("color: rgb(255, 0, 0);")
            self.label_Error.setText(
                "Reported Errors:"+"\n"+"Set Point must be between 0-1500 W")


if __name__ == '__main__':
    try:
        app = QtGui.QApplication(sys.argv)
        MyWindow = QControl(None)
        MyWindow.show()
        app.exec_()

    except rospy.ROSInterruptException:
        pass
