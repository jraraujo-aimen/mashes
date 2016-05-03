#!/usr/bin/env python
import cv2
import rospy
import math
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from mashes_measures.msg import MsgGeometry
from mashes_measures.msg import MsgVelocity
from measures.calibration import Calibration

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
        self.scale = 10
        self.size = 32
        self.calibration = Calibration(0.375)
        self.origin = (self.size * self.scale/2, self.size * self.scale/2)
        self.end = self.origin

        self.bridge = CvBridge()
        rospy.init_node('measurements_viewer')

        image_topic = rospy.get_param('~image', '/tachyon/image')
        geometry_topic = rospy.get_param('~geometry', '/tachyon/geometry')
        velocity_topic = 'velocity'

        rospy.Subscriber(image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(
            geometry_topic, MsgGeometry, self.cb_geometry, queue_size=1)
        rospy.Subscriber(
            velocity_topic, MsgVelocity, self.cb_velocity, queue_size=1)
        rospy.on_shutdown(self.on_shutdown_hook)

        cv2.namedWindow('viewer')
        cv2.cv.SetMouseCallback('viewer', self.on_mouse, '')

        rospy.spin()

    def on_shutdown_hook(self):
        cv2.destroyWindow('viewer')

    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.cv.CV_EVENT_RBUTTONDOWN:
            rospy.loginfo('New mouse event: %i, %i', x, y)

    def cb_image(self, msg_image):
        try:
            self.stamp = msg_image.header.stamp
            self.frame = self.bridge.imgmsg_to_cv2(msg_image)
            if msg_image.encoding == 'mono8':
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
            else:
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            newx, newy = self.frame.shape[1]*self.scale, self.frame.shape[0]*self.scale
            frame = cv2.resize(self.frame, (newx, newy), interpolation = cv2.INTER_NEAREST)
            image_copy = frame.copy()
            self.draw_arrow(self.speed_xy, image_copy, self.origin, self.end, self.color)
            self.draw_ellipse(self.minor_axis, self.major_axis, self.angle, self.x, self.y, image_copy, self.color)
            #cv2.ellipse(img_base_binary,(x_cm,y_cm),(length/2,width/2),rad2deg(angle),0,360,(0,255,0),1)
            cv2.imshow("viewer", image_copy)
            cv2.waitKey(1)
        except CvBridgeError, e:
            print e

    def cb_geometry(self, msg_geometry):
        self.minor_axis = self.calibration.represent(msg_geometry.minor_axis)
        self.major_axis = self.calibration.represent(msg_geometry.major_axis)
        self.angle = msg_geometry.orientation
        self.x = msg_geometry.x
        self.y = msg_geometry.y

    # velocity/speed in mm/s
    def cb_velocity(self, msg_velocity):
        self.speed = msg_velocity.speed * 1000
        self.vx = msg_velocity.vx * 1000
        self.vy = msg_velocity.vy * 1000
        self.vz = msg_velocity.vz * 1000
        speed_xy = math.sqrt(self.vx**2 + self.vy**2)
        self.speed_xy = speed_xy
        if self.speed > 0:
            self.end = (self.origin[0] + int(self.vx), self.origin[1] + int(self.vy))

    def draw_ellipse(self, minor_axis, major_axis, angle, x, y, image, color):
        if minor_axis > 0:
            print minor_axis, major_axis, angle, x, y
            center = (int(round(x*self.scale)), int(round(y*self.scale)))
            axis = (int(round(minor_axis*self.scale/2)), int(round(major_axis*self.scale/2)))
            angle_deg = np.rad2deg(angle)
            cv2.ellipse(image, center, axis, angle_deg, 0, 360, color, 1)

    def draw_arrow(self, vel, image, p, q, color, arrow_magnitude=10, thickness=2, line_type=8, shift=0):
        if vel > 0:
            if vel > 16:
                arrow_magnitude = 8
                thickness =2
            else:
                arrow_magnitude = vel*0.5
                thickness =1


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
