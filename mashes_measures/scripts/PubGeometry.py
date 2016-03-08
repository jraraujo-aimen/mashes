#!/usr/bin/env python
import rospy
import cv2
import math
from mashes_measures.msg import MsgGeometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from measures.code_geometry import Geometry


class PubGeometry():
    def __init__(self):
        rospy.init_node('pub_geometry', anonymous=True)

        image_topic = rospy.get_param('~image', '/tachyon/image')
        rospy.Subscriber(image_topic, Image, self.callback, queue_size=1)
        self.bridge = CvBridge()

        camera = image_topic.split('/')[1]
        geo_topic = '/%s/geometry' % camera
        self.geo_pub = rospy.Publisher(geo_topic, MsgGeometry, queue_size=5)
        self.msg_geo = MsgGeometry()
        self.melt_pool = Geometry()
        rospy.spin()

    def callback(self, data):
        try:
            self.stamp = data.header.stamp
            self.frame = self.bridge.imgmsg_to_cv2(data)
            if data.encoding == 'mono8':
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
            img_grey = self.melt_pool.greyscale(self.frame)
            img_bin = self.melt_pool.binarize(img_grey)
            cnt = self.melt_pool.find_contour(img_bin)
            angle_rads = 0
            major_axis = 0
            minor_axis = 0
            if cnt is not None:
                ellipse = self.melt_pool.find_ellipse(cnt)
                (x, y), (h, v), angle = ellipse
                angle_rads = math.radians(angle)
                major_axis = max(h, v)
                minor_axis = min(h, v)
                msg_text = "Angle: %s, Major: %s, Minor: %s" % (
                    angle_rads, major_axis, minor_axis)
                rospy.loginfo(msg_text)
            self.msg_geo.header.stamp = self.stamp
            self.msg_geo.orientation = angle_rads
            self.msg_geo.major_axis = major_axis
            self.msg_geo.minor_axis = minor_axis
            self.geo_pub.publish(self.msg_geo)
        except CvBridgeError, e:
            print e


if __name__ == '__main__':
    try:
        PubGeometry()
    except rospy.ROSInterruptException:
        pass
