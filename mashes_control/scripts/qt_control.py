#!/usr/bin/env python
import os
import sys
import rospy
import rospkg

from mashes_control.msg import MsgMode
from mashes_control.msg import MsgControl
from mashes_control.msg import MsgPower

from mashes_measures.msg import MsgGeometry

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore


path = rospkg.RosPack().get_path('mashes_control')


class QtControl(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        loadUi(os.path.join(path, 'resources', 'control.ui'), self)

        self.btnMode.clicked.connect(self.btnModeClicked)
        self.btnControl.clicked.connect(self.btnControlClicked)

        self.pub_mode = rospy.Publisher(
            '/control/mode', MsgMode, queue_size=10)
        self.pub_control = rospy.Publisher(
            'control/parameters', MsgControl, queue_size=10)

        self.mode = 0
        self.msg_mode = MsgMode()
        self.msg_power = MsgPower()
        self.msg_control = MsgControl()

        self.btnControlClicked()

        rospy.Subscriber('/tachyon/geometry', MsgGeometry, self.cb_geometry, queue_size=1)

    def cb_geometry(self, msg_geometry):
        self.lblInfo.setText("major_axis: %.2f" %msg_geometry.major_axis)

    def btnModeClicked(self):
        if self.mode:
            self.lblStatus.setText("Auto")
            self.lblStatus.setStyleSheet("background-color: rgb(0, 0, 255); color: rgb(255, 255, 255);")
            self.btnMode.setText("Manual")
            self.mode = 0
        else:
            self.lblStatus.setText("Manual")
            self.lblStatus.setStyleSheet("background-color: rgb(255, 0, 0); color: rgb(255, 255, 255);")
            self.btnMode.setText("Auto")
            self.mode = 1
        self.msg_mode.value = self.mode
        self.pub_mode.publish(self.msg_mode)

    def btnControlClicked(self):
        self.msg_control.set_point = self.sbSetPoint.value()
        self.msg_control.kp = self.sbKp.value()
        self.msg_control.ki = self.sbKi.value()
        self.msg_control.kd = self.sbKd.value()
        self.pub_control.publish(self.msg_control)


if __name__ == '__main__':
    rospy.init_node('control_panel')

    app = QtGui.QApplication(sys.argv)
    qt_control = QtControl()
    qt_control.show()
    app.exec_()
