#!/usr/bin/env python
import os
import tf
import sys
import cv2
import rospy
import rospkg
import numpy as np

from PyQt4 import QtCore, QtGui, uic
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scaleCamera import ScaleCamera


# Cargar nuestro archivo .ui
path = rospkg.RosPack().get_path('mashes_calibration')
form_class = uic.loadUiType(os.path.join(path, 'src/visualize', 'Scale_gui.ui'))[0]
DISPLAY_STYLE_SHEET = 'background-color: rgb(127,127,127); \
                       border: 3px solid rgb(63, 63, 63)'


def thermal_colormap(levels=1024):
    colors = np.array([[0.00, 0.00, 0.00],
                       [0.19, 0.00, 0.55],
                       [0.55, 0.00, 0.62],
                       [0.78, 0.05, 0.55],
                       [0.90, 0.27, 0.10],
                       [0.96, 0.47, 0.00],
                       [1.00, 0.70, 0.00],
                       [1.00, 0.90, 0.20],
                       [1.00, 1.00, 1.00]])
    steps = levels / (len(colors)-1)
    lut = []
    for c in range(3):
        col = []
        for k in range(1, len(colors)):
            col.append(np.linspace(colors[k-1][c], colors[k][c], steps))
        col = np.concatenate(col)
        lut.append(col)
    lut = np.transpose(np.vstack(lut))
    lut_iron = np.uint8(lut * 255)
    return lut_iron

LUT_IRON = thermal_colormap()


class QDisplay(QtGui.QWidget):
    def __init__(self, parent=None, size=64):
        super(QDisplay, self).__init__(parent)
        self.setMinimumSize(240, 240)

        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        self.setLayout(layout)

        self.lblCamera = QtGui.QLabel()
        self.lblCamera.setStyleSheet(DISPLAY_STYLE_SHEET)
        self.lblCamera.setSizePolicy(
            QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                              QtGui.QSizePolicy.Expanding))
        self.lblCamera.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(self.lblCamera)

        self.scale = 8
        self.width = size
        self.font = QtGui.QFont('Arial', self.width / 5)
        self.pixmap = QtGui.QPixmap()



    def paintFrame(self, image):
        print image.shape
        height, width = 10*image.shape[0], 10*image.shape[1]
        image = cv2.resize(image, (width, height), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Image', image)
        image = QtGui.QImage(image, width, height, QtGui.QImage.Format_RGB888)
        self.pixmap.convertFromImage(image)
        pixmap = self.pixmap.scaled(self.lblCamera.size(),
                                    QtCore.Qt.KeepAspectRatio)
        painter = QtGui.QPainter()
        painter.begin(pixmap)
        painter.setWindow(0, 0, self.scale * width, self.scale * height)
        painter.setFont(self.font)
        painter.end()
        self.lblCamera.setPixmap(pixmap)



class MyWindowClass(QtGui.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.display = QDisplay()
        self.vLayout.addWidget(self.display)

        rospy.init_node('mashes_calibration')
        self.frame_ok = False
        image_topic = rospy.get_param('~image', '/tachyon/image')
        self.bridge = CvBridge()
        rospy.Subscriber(image_topic, Image, self.cb_image, queue_size=1)


        self.shot = 0
        self.frame_stop = False
        self.btn_shot.clicked.connect(self.btn_shot_clicked)

        self.time = rospy.Time()
        self.listener = tf.TransformListener()
        self.btnQuit.clicked.connect(self.btnQuitClicked)

 # Event from btn_Mode button
    def btn_shot_clicked(self):
        self.frame_stop = True
        if self.shot == 3:
            self.shot = 0
        else:
            self.shot = self.shot + 1
        rospy.loginfo(self.shot)

        self.display.paintFrame(self.frame)
        self.process_shot()

    def cb_image(self, data):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(data)
            if data.encoding == 'mono8':
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
            if not self.frame_stop:
                self.display.paintFrame(self.frame)
            if not self.frame_ok:
                self.frame_ok = True
        except CvBridgeError, e:
            print e

    def process_shot(self):
        if self.frame_ok:
            if self.shot == 1:
                self.scale = ScaleCamera()
                #prueba
                pts_font = [[0, 0], [190, 0], [0, 190], [190, 190]]
                print pts_font
                # pts_font = np.vstack(pts_font).astype(float)
                # print pts_font
                # im_src = cv2.imread('/home/noemi/catkin_ws/src/mashes/mashes_tf/scripts/pattern_trans_2.png')
                # #real
                # #self.capture_1 = self.frame
                # #print self.capture_1
                # #prueba: Read source image.
                # self.capture_1 = cv2.imread('/home/noemi/catkin_ws/src/mashes/mashes_tf/scripts/pattern_2.png')
                #
                # print self.capture_1
                # self.cam_H_cal = self.scale.homography(im_src, pts_font, self.capture_1)
                #
                # self.listen_tf()
                # self.base_H_tool = transformations.quaternion_matrix(self.rot)
                # print self.base_H_tool.size
                # print self.base_H_tool.shape
                # self.base_H_tool[3, 0] = self.trans[0]
                # self.base_H_tool[3, 1] = self.trans[1]
                # self.base_H_tool[3, 2] = self.trans[2]
                # print "base_H_tool"
                # print self.base_H_tool
                # print "\n"

            elif self.shot == 2:
                print "rotate"
                # #real
                # #self.capture_rotate = self.frame
                # #print self.capture_rotate
                # #prueba: Read source image.
                # self.capture_rotate = cv2.imread('/home/noemi/catkin_ws/src/mashes/mashes_tf/scripts/pattern_rotate.png')
                # self.scale.offset_TCP(self.capture_rotate)

            elif self.shot == 3:
                print "moved"
                # #self.capture_moved = self.frame
                # #print self.capture_moved
                # #prueba: Read source image.
                # self.capture_moved = cv2.imread('/home/noemi/catkin_ws/src/mashes/mashes_tf/scripts/pattern_moved.png')
                # self.tool_H_cam = self.scale.turn_TCP(self.capture_moved)

    def listen_tf(self):
        self.time = rospy.Time.now()
        self.listener.waitForTransform(
            "world", "tcp0", self.time, rospy.Duration(1.0))
        (trans, rot) = self.listener.lookupTransform(
            "world", "tcp0", self.time)
        self.trans = trans
        self.rot = rot

    def btnQuitClicked(self):
        QtCore.QCoreApplication.instance().quit()

if __name__ == '__main__':
    try:
        app = QtGui.QApplication(sys.argv)
        MyWindow = MyWindowClass(None)
        MyWindow.show()
        app.exec_()

    except rospy.ROSInterruptException:
        pass
