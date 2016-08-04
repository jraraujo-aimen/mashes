#!/usr/bin/env python
import os
import rospy
import rospkg

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore

import rviz

from mashes_measures.msg import MsgVelocity
from mashes_measures.msg import MsgStatus

from qt_measures import QtMeasures


path = rospkg.RosPack().get_path('mashes_robviz')


class MyViz(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.frame = rviz.VisualizationFrame()
        self.frame.setSplashPath("")
        self.frame.initialize()

        reader = rviz.YamlConfigReader()
        config = rviz.Config()

        reader.readFile(config, os.path.join(path, 'config', 'workcell.rviz'))
        self.frame.load(config)

        self.setWindowTitle(config.mapGetChild("Title").getValue())

        self.frame.setMenuBar(None)
        self.frame.setHideButtonVisibility(False)

        self.manager = self.frame.getManager()
        self.grid_display = self.manager.getRootDisplayGroup().getDisplayAt(0)

        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(9, 0, 9, 0)
        self.setLayout(layout)

        h_layout = QtGui.QHBoxLayout()
        layout.addLayout(h_layout)

        orbit_button = QtGui.QPushButton("Orbit View")
        orbit_button.clicked.connect(self.onOrbitButtonClick)
        h_layout.addWidget(orbit_button)

        front_button = QtGui.QPushButton("Front View")
        front_button.clicked.connect(self.onFrontButtonClick)
        h_layout.addWidget(front_button)

        right_button = QtGui.QPushButton("Rigth View")
        right_button.clicked.connect(self.onRightButtonClick)
        h_layout.addWidget(right_button)

        top_button = QtGui.QPushButton("Top View")
        top_button.clicked.connect(self.onTopButtonClick)
        h_layout.addWidget(top_button)

        layout.addWidget(self.frame)

    def switchToView(self, view_name):
        view_man = self.manager.getViewManager()
        for i in range(view_man.getNumViews()):
            if view_man.getViewAt(i).getName() == view_name:
                view_man.setCurrentFrom(view_man.getViewAt(i))
                return
        print("Did not find view named %s." % view_name)

    def onOrbitButtonClick(self):
        self.switchToView("Orbit View")

    def onFrontButtonClick(self):
        self.switchToView("Front View")

    def onRightButtonClick(self):
        self.switchToView("Right View")

    def onTopButtonClick(self):
        self.switchToView("Top View")


class Robviz(QtGui.QMainWindow):
    def __init__(self):
        super(Robviz, self).__init__()
        loadUi(os.path.join(path, 'resources', 'robviz.ui'), self)

        self.boxPlot.addWidget(MyViz())

        self.qtMeasures = QtMeasures()
        self.tabWidget.addTab(self.qtMeasures, 'Measures')
        self.tabWidget.setCurrentWidget(self.qtMeasures)

        self.btnQuit.clicked.connect(self.btnQuitClicked)

        rospy.Subscriber('/velocity', MsgVelocity, self.cb_velocity, queue_size=1)
        rospy.Subscriber('/supervisor/status', MsgStatus, self.cb_status, queue_size=1)

    def cb_velocity(self, msg_velocity):
        self.lblInfo.setText("Speed: %.1f mm/s" % (1000 * msg_velocity.speed))

    def cb_status(self, msg_status):
        txt_status = ''
        if msg_status.laser_on:
            txt_status = 'Laser ON' + '\n'
            # self.lblStatus.setStyleSheet(
            #     "background-color: rgb(255, 255, 0); color: rgb(0, 0, 0);")
        else:
            txt_status = 'Laser OFF' + '\n'
            # self.lblStatus.setStyleSheet(
            #     "background-color: rgb(255, 255, 0); color: rgb(0, 0, 0);")
        if msg_status.running:
            txt_status = txt_status + 'Running'
        else:
            txt_status = txt_status + 'Stopped'
        self.lblStatus.setText(txt_status)

    def btnQuitClicked(self):
        QtCore.QCoreApplication.instance().quit()


if __name__ == '__main__':
    import sys

    rospy.init_node('robviz')

    app = QtGui.QApplication(sys.argv)
    robviz = Robviz()
    robviz.show()
    sys.exit(app.exec_())
