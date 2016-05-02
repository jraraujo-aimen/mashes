#!/usr/bin/env python
import tf
import rospy
import numpy as np

from mashes_measures.msg import MsgVelocity
from measures.velocity import Velocity


class NdVelocity():
    def __init__(self):
        rospy.init_node('velocity')
        velocity_topic = 'velocity'
        self.velocity_pub = rospy.Publisher(
            velocity_topic, MsgVelocity, queue_size=10)

        self.velocity = Velocity()
        self.msg_velocity = MsgVelocity()
        self.listener = tf.TransformListener()

        self.time = rospy.Time()
        self.prev_time = rospy.Time()
        r = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            try:
                rospy.loginfo("pub_velocity")
                self.pub_velocity()
            except:
                rospy.logerr("")
            r.sleep()

    def pub_velocity(self):
        # Make sure we see the world and tcp frames
        self.listener.waitForTransform(
            "/world", "/tcp0", rospy.Time(), rospy.Duration(5.0))
        try:
            stamp = rospy.Time.now()
            self.time = stamp
            self.listener.waitForTransform(
                "/world", "/tcp0", stamp, rospy.Duration(1.0))
            position, quaternion = self.listener.lookupTransform(
                "/world", "/tcp0", stamp)
            #velocity in m/s
            v_vector3, speed = self.velocity.instantaneous_vector(
                stamp, np.array(position))

            v_vector3_camera = self.listener.transformVector3(
                "camera0", v_vector3)

            self.msg_velocity.header.stamp = stamp
            self.msg_velocity.speed = speed
            self.msg_velocity.vx = v_vector3_camera.vector.x
            self.msg_velocity.vy = v_vector3_camera.vector.y
            self.msg_velocity.vz = v_vector3_camera.vector.z
            rospy.loginfo(stamp.to_sec())
            rospy.loginfo(speed)
            rospy.loginfo(self.msg_velocity)
            self.velocity_pub.publish(self.msg_velocity)
            rospy.loginfo("published")

        except (tf.Exception, tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            rospy.loginfo("TF Exception")

if __name__ == '__main__':
    try:
        NdVelocity()
    except rospy.ROSInterruptException:
        pass
