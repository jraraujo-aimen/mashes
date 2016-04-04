#!/usr/bin/env python
import cv2
import rospy
from mashes_measures.msg import MsgGeometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from measures.geometry import Geometry


class PubGeometry():
    def __init__(self):
        rospy.init_node('pub_geometry', anonymous=True)

        image_topic = rospy.get_param('~image', '/tachyon/image')
        rospy.Subscriber(image_topic, Image, self.callback, queue_size=1)
        self.bridge = CvBridge()

        geo_topic = '/%s/geometry' % image_topic.split('/')[1]
        self.pub_geo = rospy.Publisher(geo_topic, MsgGeometry, queue_size=10)
        self.msg_geo = MsgGeometry()
        self.geometry = Geometry()

        rospy.spin()

    def callback(self, data):
        try:
            stamp = data.header.stamp
            frame = self.bridge.imgmsg_to_cv2(data)
            if data.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            major_axis, minor_axis, angle = self.geometry.find_geometry(frame)
            self.msg_geo.header.stamp = stamp
            self.msg_geo.major_axis = major_axis
            self.msg_geo.minor_axis = minor_axis
            self.msg_geo.orientation = angle
            self.pub_geo.publish(self.msg_geo)
        except CvBridgeError, e:
            print e


if __name__ == '__main__':
    try:
        PubGeometry()
    except rospy.ROSInterruptException:
        pass
