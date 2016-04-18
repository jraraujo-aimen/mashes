import numpy as np


class Velocity():
    def __init__(self):
        self.time = None
        self.position = None

    def instantaneous(self, time, position):
        if self.position is None:
            speed = 0
        else:
            dt = time - self.time
            dp = position - self.position
            speed = np.sqrt(np.sum(dp * dp)) / dt
        self.time = time
        self.position = position
        if speed < 0.0005:
            speed = 0.0
        return np.around(speed, decimals=4)


if __name__ == '__main__':
    t1 = 1448535428.73
    p1 = np.array([1.64148, 0.043086, 0.944961])
    q1 = np.array([0.00566804, 0.000861386, -0.0100175, 0.999933])
    t2 = 1448535428.75
    p2 = np.array([1.64148, 0.043865, 0.944964])
    q2 = np.array([0.00566161, 0.000860593, -0.0100132, 0.999933])
    t3 = 1448535429.22
    p3 = np.array([1.64148, 0.047131, 0.944964])
    q3 = np.array([0.00566494, 0.000858606, -0.0100118, 0.999933])

    velocity = Velocity()
    print velocity.instantaneous(t1, p1)
    print velocity.instantaneous(t2, p2)
    print velocity.instantaneous(t3, p3)
