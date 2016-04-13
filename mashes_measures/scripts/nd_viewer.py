#!/usr/bin/env python
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from mashes_measures.msg import MsgGeometry


class NdViewer():
    def __init__(self):
        rospy.init_node('measurements_viewer')

        image_topic = rospy.get_param('~image', '/tachyon/image')
        geometry_topic = rospy.get_param('~geometry', '/tachyon/geometry')

        rospy.Subscriber(image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(geometry_topic, MsgGeometry, self.cb_geometry, queue_size=1)
        rospy.on_shutdown(self.on_shutdown_hook)

        self.bridge = CvBridge()

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
            frame = cv2.resize(self.frame, (256, 256))
            #cv2.ellipse(img_base_binary,(x_cm,y_cm),(length/2,width/2),rad2deg(angle),0,360,(0,255,0),1)
            cv2.imshow("viewer", frame)
            cv2.waitKey(1)
        except CvBridgeError, e:
            print e

    def cb_geometry(self, msg_geometry):
        self.minor_axis = msg_geometry.minor_axis
        self.major_axis = msg_geometry.major_axis
        self.angle = msg_geometry.orientation


if __name__ == '__main__':
    try:
        NdViewer()
    except rospy.ROSInterruptException:
        pass
