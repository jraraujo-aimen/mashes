#!/usr/bin/env python
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from mashes_measures.msg import MsgGeometry
from measures.moments import Moments


class NdMoments():
    def __init__(self):
        rospy.init_node('moments')

        image_topic = rospy.get_param('~image', '/tachyon/image')
        rospy.Subscriber(image_topic, Image, self.callback, queue_size=1)
        self.bridge = CvBridge()

        geo_topic = '/%s/moments' % image_topic.split('/')[1]
        self.pub_geo = rospy.Publisher(geo_topic, MsgGeometry, queue_size=10)
        self.msg_geo = MsgGeometry()

        threshold = rospy.get_param('~threshold', 127)
        self.moments = Moments(threshold)

        rospy.spin()

    def callback(self, data):
        try:
            stamp = data.header.stamp
            frame = self.bridge.imgmsg_to_cv2(data)
            if data.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            major_axis, minor_axis, angle = self.moments.find_geometry(frame)
            self.msg_geo.header.stamp = stamp
            self.msg_geo.major_axis = major_axis
            self.msg_geo.minor_axis = minor_axis
            self.msg_geo.orientation = angle
            self.pub_geo.publish(self.msg_geo)
        except CvBridgeError, e:
            print e


if __name__ == '__main__':
    try:
        NdMoments()
    except rospy.ROSInterruptException:
        pass


#             if (length >0) and (width >0):
#                 x         =x_cm
#                 y         =y_cm
#                 major_16  = np.uint16(major_axis)
#                 minor_16  = np.uint16(minor_axis)
#                 angle_16  = np.uint16(np.rad2deg(angle_rads))
#                 x_16      = np.uint16(x_cm)
#                 y_16      = np.uint16(y_cm)
#                 cv2.ellipse(color,(x_16,y_16), (major_16,minor_16),angle_16,0,360,(0,255,0),1)
#                 print"PixelIntensity : Angle: %s, Major: %s, Minor: %s, x: %s, y: %s" % (
#                     angle_rads, major_axis, minor_axis, x ,y)
#
#             cv2.imshow("Pixel_intensity",cv2.resize(color,(200,200)))
#             cv2.waitKey(3)
