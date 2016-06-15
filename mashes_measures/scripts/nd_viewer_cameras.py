#!/usr/bin/env python
import os
import cv2
import math
import rospy
import rospkg
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from mashes_measures.msg import MsgGeometryViewer
from mashes_measures.msg import MsgVelocity
from measures.calibration import Calibration
from measures.projection import Projection


class NdViewer():
    def __init__(self):
        self.color = (0, 0, 0)
        self.speed = 0
        self.speed_xy = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.minor_axis = 0
        self.major_axis = 0
        self.angle = 0
        self.x = 0
        self.y = 0
        # self.scale = 20
        # self.size = 32
        self.calibration = Calibration(0.375)
        self.TCP = np.float32([[0, 0]])

        # self.origin = (self.size * self.scale/2, self.size * self.scale/2)
        # self.end = self.origin

        self.bridge = CvBridge()
        rospy.init_node('viewer_cameras')

        path = rospkg.RosPack().get_path('mashes_measures')

        self.p_uEye = Projection()
        self.p_uEye.load_configuration(
            os.path.join(path, 'config/uEye_config.yaml'))
        self.p_tachyon = Projection()
        self.p_tachyon.load_configuration(
            os.path.join(path, 'config/NIT_config.yaml'))

        pxl_origin = self.p_uEye.transform(self.p_uEye.hom, self.TCP)
        visual_origin = self.p_uEye.transform(self.p_uEye.hom_vis, pxl_origin)
        self.origin = (int(visual_origin[0][0]), int(visual_origin[0][1]))
        self.end = self.origin

        rospy.Subscriber(
            '/camera/image', Image, self.cb_image_camera, queue_size=1)
        rospy.Subscriber(
            '/tachyon/image', Image, self.cb_image_tachyon, queue_size=1)
        rospy.Subscriber(
            '/camera/geometry_viewer', MsgGeometryViewer, self.cb_geometry, queue_size=1)
        rospy.Subscriber(
            'velocity', MsgVelocity, self.cb_velocity, queue_size=1)

        self.pub_image = rospy.Publisher('/measures/image', Image, queue_size=10)

        self.frame_uEye = None
        self.frame_tachyon = None

        r = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            try:
                self.paint_images()
            except:
                rospy.logerr("paint images")
            r.sleep()

    def cb_image_camera(self, msg_image):
        try:
            self.stamp1 = msg_image.header.stamp
            self.frame_uEye = self.bridge.imgmsg_to_cv2(msg_image)
            if msg_image.encoding == 'mono8':
                self.frame_uEye = cv2.cvtColor(
                    self.frame_uEye, cv2.COLOR_GRAY2BGR)
            else:
                self.frame_uEye = cv2.cvtColor(
                    self.frame_uEye, cv2.COLOR_RGB2BGR)
        except CvBridgeError, e:
            print e

    def cb_image_tachyon(self, msg_image):
        try:
            self.stamp = msg_image.header.stamp
            self.frame_tachyon = self.bridge.imgmsg_to_cv2(msg_image)
            if msg_image.encoding == 'mono8':
                self.frame_tachyon = cv2.cvtColor(
                    self.frame_tachyon, cv2.COLOR_GRAY2BGR)
            else:
                self.frame_tachyon = cv2.cvtColor(
                    self.frame_tachyon, cv2.COLOR_RGB2BGR)
        except CvBridgeError, e:
            print e

    def paint_images(self):
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        print image.shape
        if self.frame_uEye is not None:
            im_camera = self.p_uEye.project_image(
                self.frame_uEye, self.p_uEye.hom_vis)
            im_uEye = self.p_uEye.draw_point(im_camera, self.TCP)
            print im_uEye.shape
            image = cv2.addWeighted(image, 1, im_uEye, 0.4, 0)
        if self.frame_tachyon is not None:
            im_tachyon = self.p_tachyon.project_image(
                self.frame_tachyon, self.p_tachyon.hom_vis)
            im_tachyon = self.p_tachyon.draw_point(im_tachyon, self.TCP)
            print im_tachyon.shape
            image = cv2.addWeighted(image, 1, im_tachyon, 0.6, 0)
        self.draw_arrow(
            self.speed_xy, image, self.origin, self.end, self.color)
        self.draw_ellipse(
            self.minor_axis, self.major_axis, self.angle, self.x, self.y,
            image, self.color)
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

    def cb_geometry(self, msg_geometry_view):

        self.minor_axis = msg_geometry_view.minor_axis
        self.major_axis = msg_geometry_view.major_axis
        self.angle = msg_geometry_view.orientation
        self.x = msg_geometry_view.x
        self.y = msg_geometry_view.y

    # velocity/speed in mm/s
    def cb_velocity(self, msg_velocity):

        self.speed = msg_velocity.speed * 1000
        self.vx = msg_velocity.vx * 1000
        self.vy = msg_velocity.vy * 1000
        self.vz = msg_velocity.vz * 1000
        speed_xy = math.sqrt(self.vx**2 + self.vy**2)
        self.speed_xy = speed_xy
        if self.speed > 0:
            self.end = (
                self.origin[0] + int(self.vx*3), self.origin[1] + int(self.vy*3))

    def draw_ellipse(self, minor_axis, major_axis, angle, x, y, image, color):

        if minor_axis > 0:
            center = (int(x), int(y))
            axis = (int(round(minor_axis/2)), int(round(major_axis/2)))
            angle_deg = np.rad2deg(angle)
            cv2.ellipse(image, center, axis, angle_deg, 0, 360, color, 1)

    def draw_arrow(self, vel, image, p, q, color, arrow_magnitude=10,
                   thickness=2, line_type=8, shift=0):

        if vel > 0:
            if vel > 16:
                arrow_magnitude = 24
                thickness = 2
            else:
                arrow_magnitude = vel*1.2
                thickness = 1

            # draw arrow tail
            cv2.line(image, p, q, color, thickness, line_type, shift)
            # calc angle of the arrow
            angle = np.arctan2(p[1]-q[1], p[0]-q[0])
            # starting point of first line of arrow head
            p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
                 int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
            # draw first half of arrow head
            cv2.line(image, p, q, color, thickness, line_type, shift)
            # starting point of second line of arrow head
            p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
                 int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
            # draw second half of arrow head
            cv2.line(image, p, q, color, thickness, line_type, shift)


if __name__ == '__main__':
    try:
        NdViewer()
    except rospy.ROSInterruptException:
        pass
