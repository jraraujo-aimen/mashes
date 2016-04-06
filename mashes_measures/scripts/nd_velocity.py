#!/usr/bin/env python
import tf
import rospy
import numpy as np

from mashes_measures.msg import MsgVelocity
from measures.velocity import Velocity


class NdVelocity():
    def __init__(self):
        rospy.init_node('velocity')

        self.velocity_pub = rospy.Publisher(
            'velocity', MsgVelocity, queue_size=5)

        self.velocity = Velocity()
        self.msg_velocity = MsgVelocity()
        self.listener = tf.TransformListener()

        r = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            try:
                self.pub_velocity()
            except:
                rospy.logerr("")
            r.sleep()

    def pub_velocity(self):
        stamp = rospy.Time.now()
        self.listener.waitForTransform("/world", "/tcp0", stamp, rospy.Duration(1.0))
        position, quaternion = self.listener.lookupTransform("/world", "/tcp0", stamp)
        #matrix = tf.transformations.quaternion_matrix(quaternion)
        speed = self.velocity.instantaneous(stamp.to_sec(), np.array(position))
        self.msg_velocity.header.stamp = stamp
        self.msg_velocity.speed = speed
        rospy.loginfo(stamp.to_sec())
        rospy.loginfo(speed)
        self.velocity_pub.publish(self.msg_velocity)


if __name__ == '__main__':
    try:
        NdVelocity()
    except rospy.ROSInterruptException:
        pass
