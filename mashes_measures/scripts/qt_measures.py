#!/usr/bin/env python
import os
import sys
import rospy
import rospkg

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore

from mashes_tachyon.msg import MsgCalibrate


class QtMeasures(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        path = rospkg.RosPack().get_path('mashes_measures')
        loadUi(os.path.join(path, 'resources', 'measures.ui'), self)

        self.btnCalibrate.clicked.connect(self.btnCalibrateClicked)
        self.btnRecord.clicked.connect(self.btnRecordClicked)

        self.pub_calibrate = rospy.Publisher(
            '/tachyon/calibrate', MsgCalibrate, queue_size=10)
        self.msg_calibrate = MsgCalibrate()

        self.recording = False

    def btnCalibrateClicked(self):
        self.msg_calibrate.calibrate = 1
        self.pub_calibrate.publish(self.msg_calibrate)

    def btnRecordClicked(self):
        if self.recording:
            self.recording = False
            self.btnRecord.setText('Record cloud')
        else:
            self.recording = True
            self.btnRecord.setText('Stop recording...')


if __name__ == "__main__":
    rospy.init_node('measures_panel')

    app = QtGui.QApplication(sys.argv)
    qt_measures = QtMeasures()
    qt_measures.show()
    app.exec_()
