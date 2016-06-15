import numpy as np
import cv2


class CoolRate():
    def __init__(self):
        self.time = None
        self.dt = None
        self.ds = None

    def instantaneous(self, time, vel):
        if self.time is not None:
            dt = time - self.time
            self.dt = dt
            ds = vel * dt
            self.ds = ds
        self.time = time

    def position(self, position):
        if self.dt is not None:
            #position in mm
            position_0 = position + self.ds
            return position_0
        else:
            return None

if __name__ == '__main__':

    coolrate = CoolRate()

    frame1 = cv2.imread("../../data/tachyon/frame0000.jpg")
    time1 = 1462534491.31
    position1 = np.array([1.76853151, -0.17615799,  1.02817866])
    speed1 = 0.00814883577822
    vel1 = np.array([0.0004,  0.0081,  0.0006])
    coolrate.instantaneous(time1, vel1)
    p1 = coolrate.position(position1)
    print "Data:", position1, p1

    frame2 = cv2.imread("../../data/tachyon/frame0001.jpg")
    time2 = 1462534491.42
    position2 = np.array([1.76847009, -0.1751599,   1.0282047])
    speed2 = 0.00873904550319
    vel2 = np.array([-0.0005,  0.0087,  0.0002])
    coolrate.instantaneous(time2, vel2)
    p2 = coolrate.position(position2)
    print "Data:", position2, p2

    frame3 = cv2.imread("../../data/tachyon/frame0002.jpg")
    time3 = 1462534491.51
    position3 = np.array([1.76854256, -0.1745629,   1.02820809])
    speed3 = 0.00705102882895
    vel3 = np.array([0.0008,  0.007,   0.])
    coolrate.instantaneous(time3, vel3)
    p3 = coolrate.position(position3)
    print "Data:", position3, p3

    frame4 = cv2.imread("../../data/tachyon/frame0003.jpg")
    time4 = 1462534491.61
    position4 = np.array([1.76842354, -0.17369612,  1.02805208])
    speed4 = 0.00886784776371
    vel4 = np.array([-0.0012,  0.0086, -0.0016])
    coolrate.instantaneous(time4, vel4)
    p4 = coolrate.position(position4)
    print "Data:", position4, p4

    frame5 = cv2.imread("../../data/tachyon/frame0004.jpg")
    time5 = 1462534491.71
    position5 = np.array([1.76853454, -0.17285232,  1.02808612])
    speed5 = 0.0085360744843
    vel5 = np.array([0.0011,  0.0085,  0.0003])
    coolrate.instantaneous(time5, vel5)
    p5 = coolrate.position(position5)
    print "Data:", position5, p5
