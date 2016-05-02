import numpy as np
from geometry_msgs.msg import Vector3Stamped, Vector3
import rospy


class Velocity():
    def __init__(self):
        self.time = None
        self.position = None

    ##create a Vector3Stamped message
    def create_vector3_stamped(self, vector, time, frame_id="world"):
        m = Vector3Stamped()
        m.header.frame_id = frame_id
        m.header.stamp = time
        m.vector = Vector3(*vector)
        return m

    def instantaneous_vector(self, time, position):
        if self.position is None:
            speed = 0
            v = (0, 0, 0)
        else:
            # pos = np.array([position[0], position[1], position[2]])
            dt = time - self.time
            dt_nsecs = dt.nsecs
            dp = position - self.position
            speed = (np.sqrt(np.sum(dp * dp)) / dt_nsecs)*1000000000

            v_x = (dp[0]/dt_nsecs)*1000000000
            v_y = (dp[1]/dt_nsecs)*1000000000
            v_z = (dp[2]/dt_nsecs)*1000000000

            if v_x < 0.0005:
                v_x = 0.0
            v_x = np.around(v_x, decimals=4)

            if v_y < 0.0005:
                v_y = 0.0
            v_y = np.around(v_y, decimals=4)

            if v_z < 0.0005:
                v_z = 0.0
            v_z = np.around(v_z, decimals=4)

            v = (v_x, v_y, v_z)

        if speed < 0.0005:
            speed = 0.0
        self.time = time
        # self.position = np.array([position[0], position[1], position[2]])
        self.position = position
        v_vector3 = self.create_vector3_stamped(v, time)
        return v_vector3, np.around(speed, decimals=4)


if __name__ == '__main__':
    t1 = rospy.Time(1448535428.73)
    p1 = np.array([1.64148, 0.043086, 0.944961])
    q1 = np.array([0.00566804, 0.000861386, -0.0100175, 0.999933])
    rospy.sleep(0.1)
    t2 = rospy.Time(1448535428.75)
    p2 = np.array([1.64148, 0.043865, 0.944964])
    q2 = np.array([0.00566161, 0.000860593, -0.0100132, 0.999933])
    rospy.sleep(0.1)
    t3 = rospy.Time(1448535429.22)
    p3 = np.array([1.64148, 0.047131, 0.944964])
    q3 = np.array([0.00566494, 0.000858606, -0.0100118, 0.999933])

    velocity = Velocity()
    print "...First"
    vector, speed = velocity.instantaneous_vector(t1, p1)
    print vector
    print "      "
    print "...Second"
    vector, speed = velocity.instantaneous_vector(t2, p2)
    print vector
    print "      "
    print "...Third"
    vector, speed = velocity.instantaneous_vector(t3, p3)
    print vector
