#!/usr/bin/env python
import os
import cv2
import math
import rospy
import rospkg
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from mashes_measures.msg import MsgGeometry
from mashes_measures.msg import MsgGeometryViewer
from measures.geometry import Geometry
from measures.projection import Projection


class NdGeometry():
    def __init__(self):
        rospy.init_node('geometry')
        path = rospkg.RosPack().get_path('mashes_measures')

        self.p_uEye = Projection()
        self.p_uEye.load_configuration(
            os.path.join(path, 'config/uEye_config.yaml'))

        #image_topic = rospy.get_param('~image', '/tachyon/image')
        image_topic = rospy.get_param('~image', '/camera/image')
        rospy.Subscriber(image_topic, Image, self.cb_image, queue_size=1)
        self.bridge = CvBridge()

        geo_topic = '/%s/geometry' % image_topic.split('/')[1]
        self.pub_geo = rospy.Publisher(geo_topic, MsgGeometry, queue_size=10)
        self.msg_geo = MsgGeometry()

        geo_view_topic = '/%s/geometry_viewer' % image_topic.split('/')[1]
        self.pub_geo_view = rospy.Publisher(
            geo_view_topic, MsgGeometryViewer, queue_size=10)
        self.msg_geo_view = MsgGeometryViewer()

        threshold = rospy.get_param('~threshold', 127)
        self.geometry = Geometry(threshold)

        rospy.spin()

    def cb_image(self, msg_image):
        try:
            stamp = msg_image.header.stamp
            frame = self.bridge.imgmsg_to_cv2(msg_image)
            if msg_image.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            im_camera = self.p_uEye.project_image(frame, self.p_uEye.hom_vis)
            mj_axis_vis, mn_axis_vis, a_vis, c_vis = self.geometry.find_geometry(im_camera)

            self.msg_geo_view.header.stamp = stamp
            self.msg_geo_view.major_axis = mj_axis_vis
            self.msg_geo_view.minor_axis = mn_axis_vis
            self.msg_geo_view.x = c_vis[0]
            self.msg_geo_view.y = c_vis[1]
            self.msg_geo_view.orientation = a_vis

            self.pub_geo_view.publish(self.msg_geo_view)

            major_0, major_1, minor_0, minor_1 = self.get_pixels(mj_axis_vis, mn_axis_vis, a_vis, c_vis)
            major = np.float32([major_0, major_1])
            minor = np.float32([minor_0, minor_1])
            center = np.float32([[c_vis[0], c_vis[1]]])
            pnt_major, pnt_minor, pnt_center = self.get_points(major, minor, center)
            d_major = self.get_distance(pnt_major[0], pnt_major[1])
            d_minor = self.get_distance(pnt_minor[0], pnt_minor[1])

            self.msg_geo.header.stamp = stamp
            self.msg_geo.major_axis = d_major
            self.msg_geo.minor_axis = d_minor
            self.msg_geo.x = pnt_center[0][0]
            self.msg_geo.y = pnt_center[0][1]
            self.msg_geo.orientation = a_vis

            self.pub_geo.publish(self.msg_geo)
        except CvBridgeError, e:
            print e



    def get_distance(self, pnt_1, pnt_2):
        d = math.sqrt((pnt_2[0]-pnt_1[0])**2 + (pnt_2[1]-pnt_1[1])**2)

        return d

    def get_points(self, major, minor, center):
        pxl_major = self.p_uEye.transform(self.p_uEye.inv_hom_vis, major)
        pnt_major = self.p_uEye.transform(self.p_uEye.inv_hom, pxl_major)
        pxl_minor = self.p_uEye.transform(self.p_uEye.inv_hom_vis, minor)
        pnt_minor = self.p_uEye.transform(self.p_uEye.inv_hom, pxl_minor)
        pxl_center = self.p_uEye.transform(self.p_uEye.inv_hom_vis, center)
        pnt_center = self.p_uEye.transform(self.p_uEye.inv_hom, pxl_center)

        return pnt_major, pnt_minor, pnt_center

    def get_pixels(self, mj, mn, a, c):
        #major
        major_0 = [c[0] + mj*math.cos(a), c[1] - mj*math.sin(a)]
        major_1 = [c[0] - mj*math.cos(a), c[1] + mj*math.sin(a)]

        #minor
        minor_0 = [c[0] - mn*math.cos((math.pi)/2 - a), c[1] - mn*math.sin((math.pi)/2 - a)]
        minor_1 = [c[0] + mn*math.cos((math.pi)/2 - a), c[1] + mn*math.sin((math.pi)/2 - a)]

        return major_0, major_1, minor_0, minor_1

if __name__ == '__main__':
    try:
        NdGeometry()
    except rospy.ROSInterruptException:
        pass
