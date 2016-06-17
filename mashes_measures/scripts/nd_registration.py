#!/usr/bin/env python
import os
import cv2
import rospy
import rospkg
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from mashes_measures.msg import MsgGeometry
from mashes_measures.msg import MsgVelocity

from measures.projection import Projection


class NdRegistration():
    def __init__(self):
        rospy.init_node('registration')

        # Geometry
        self.tachyon_ellipse = (0, 0), (0, 0), 0
        self.camera_ellipse = (0, 0), (0, 0), 0

        # Speed
        self.speed = 0
        self.vel = (0, 0, 0)

        self.bridge = CvBridge()

        path = rospkg.RosPack().get_path('mashes_measures')
        self.p_camera = Projection()
        self.p_camera.load_configuration(
            os.path.join(path, 'config', 'camera.yaml'))
        self.p_tachyon = Projection()
        self.p_tachyon.load_configuration(
            os.path.join(path, 'config', 'tachyon.yaml'))

        rospy.Subscriber('/tachyon/image', Image,
                         self.cb_image_tachyon, queue_size=1)
        rospy.Subscriber('/camera/image', Image,
                         self.cb_image_camera, queue_size=1)
        rospy.Subscriber('/tachyon/geometry', MsgGeometry,
                         self.cb_geometry_tachyon, queue_size=1)
        rospy.Subscriber('/camera/geometry', MsgGeometry,
                         self.cb_geometry_camera, queue_size=1)
        rospy.Subscriber('/velocity', MsgVelocity,
                         self.cb_velocity, queue_size=1)

        self.pub_image = rospy.Publisher('/measures/image', Image, queue_size=10)

        self.frame_camera = None
        self.frame_tachyon = None

        r = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            try:
                self.paint_images()
            except:
                rospy.logerr("paint images")
            r.sleep()

    def cb_image_tachyon(self, msg_image):
        try:
            self.stamp = msg_image.header.stamp
            self.frame_tachyon = self.bridge.imgmsg_to_cv2(msg_image)
        except CvBridgeError, e:
            print e

    def cb_image_camera(self, msg_image):
        try:
            self.stamp1 = msg_image.header.stamp
            self.frame_camera = self.bridge.imgmsg_to_cv2(msg_image)
        except CvBridgeError, e:
            print e

    def cb_geometry_tachyon(self, msg_geometry):
        self.tachyon_ellipse = ((msg_geometry.x, msg_geometry.y),
                                (msg_geometry.major_axis, msg_geometry.minor_axis),
                                msg_geometry.orientation)

    def cb_geometry_camera(self, msg_geometry):
        self.camera_ellipse = ((msg_geometry.x, msg_geometry.y),
                               (msg_geometry.major_axis, msg_geometry.minor_axis),
                               msg_geometry.orientation)

    def cb_velocity(self, msg_velocity):
        # velocity/speed in mm/s
        self.speed = msg_velocity.speed * 1000
        self.vel = (msg_velocity.vx * 1000,
                    msg_velocity.vy * 1000,
                    msg_velocity.vz * 1000)

    def paint_images(self):
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        if self.frame_camera is not None:
            img_camera = cv2.cvtColor(self.frame_camera, cv2.COLOR_GRAY2BGR)
            img_camera = self.p_camera.project_image(img_camera)
            img_camera = self.p_camera.draw_TCP_axis(img_camera)
            img_camera = self.p_camera.draw_ellipse(
                img_camera, self.camera_ellipse)
            image = cv2.addWeighted(image, 1, img_camera, 0.4, 0)
        if self.frame_tachyon is not None:
            img_tachyon = cv2.cvtColor(self.frame_tachyon, cv2.COLOR_RGB2BGR)
            img_tachyon = self.p_tachyon.project_image(img_tachyon)
            img_tachyon = self.p_tachyon.draw_TCP_axis(img_tachyon)
            img_tachyon = self.p_tachyon.draw_ellipse(
                img_tachyon, self.tachyon_ellipse)
            image = cv2.addWeighted(image, 1, img_tachyon, 0.6, 0)
        self.p_tachyon.draw_arrow(image, self.speed, self.vel)
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))


if __name__ == '__main__':
    try:
        NdRegistration()
    except rospy.ROSInterruptException:
        pass
