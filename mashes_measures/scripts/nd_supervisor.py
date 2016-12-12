#!/usr/bin/env python
import rospy

from mashes_measures.msg import MsgGeometry
from mashes_measures.msg import MsgVelocity
from mashes_measures.msg import MsgStatus


class NdSupervisor():
    def __init__(self):
        rospy.init_node('supervisor')

        self.pub_status = rospy.Publisher(
            '/supervisor/status', MsgStatus, queue_size=10)

        rospy.Subscriber(
            '/tachyon/geometry', MsgGeometry, self.cb_geometry, queue_size=1)
        rospy.Subscriber(
            '/velocity', MsgVelocity, self.cb_velocity, queue_size=1)

        self.msg_status = MsgStatus()
        self.msg_status.laser_on = False
        self.msg_status.running = False

        r = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            self.cb_status()
            r.sleep()

    def cb_status(self):
        #stamp = rospy.Time.now()
        self.pub_status.publish(self.msg_status)

    def cb_geometry(self, msg_geometry):
        power = 0
        laser_on = False
        if msg_geometry.minor_axis > 0.5:
            laser_on = True
            power = rospy.get_param('/process/power')
        self.msg_status.laser_on = laser_on
        self.msg_status.power = power

    def cb_velocity(self, msg_velocity):
        speed = 0
        running = False
        if msg_velocity.speed > 0.0005:
            running = True
            speed = 1000 * msg_velocity.speed
        self.msg_status.running = running
        self.msg_status.speed = speed


if __name__ == '__main__':
    try:
        NdSupervisor()
    except rospy.ROSInterruptException:
        pass
