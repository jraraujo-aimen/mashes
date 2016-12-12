import numpy as np


class Angle():
    def __init__(self):
        self.angle = 0

    def director_vector(self, pt1, pt2):
        vd = (pt2[0]-pt1[0], pt2[1]-pt1[1])
        return vd

    def calculate_angle(self, vd1, vd2):
        den = abs(vd1[0]*vd2[0]+vd1[1]*vd2[1])
        num1 = np.sqrt(vd1[0]**2+vd1[1]**2)
        num2 = np.sqrt(vd2[0]**2+vd2[1]**2)
        cos_angle = den/(num1*num2)
        angle = np.rad2deg(np.arccos(cos_angle))
        return angle


if __name__ == '__main__':
    a = Angle()
    l_1 = np.float32([[200, 50], [200, 150]])
    l_2 = np.float32([[400, 45], [420, 280]])
    vd1 = a.director_vector(l_1[0], l_1[1])
    vd2 = a.director_vector(l_2[0], l_2[1])
    print vd1, vd2
    angle = a.calculate_angle(vd1, vd2)
    print angle
