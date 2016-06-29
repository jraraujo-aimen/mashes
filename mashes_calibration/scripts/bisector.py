import numpy as np

class Perp_bisector(object):
    def __init__(self, pto_ant, pto_post):
        (self.x1, self.y1) = pto_ant
        (self.x2, self.y2) = pto_post
        self.calculate_midpoint()
        self.calculate_slop()
        self.calculate_coef()


    def calculate_midpoint(self):
        xp = (self.x2+self.x1)/2
        yp = (self.y2+self.y1)/2
        self.midpoint = (xp, yp)

    def calculate_slop(self):
        slop = (self.x1 - self.x2)/(self.y2 - self.y1)
        self.slop = slop

    def calculate_coef(self):
        #y=AX+B
        (xp, yp) = self.midpoint
        self.A = self.slop
        self.B = yp - self.slop*xp


class Intersection(object):
    def __init__(self, bisector_1, bisector_2):
        self.bisector_1 = bisector_1
        self.bisector_2 = bisector_2
        self.calculate_x()
        self.calculate_y()

    def calculate_x(self):
        self.x = (self.bisector_2.B - self.bisector_1.B)/(self.bisector_1.A - self.bisector_2.A)

    def calculate_y(self):
        self.y = self.bisector_1.A * self.x + self.bisector_1.B

if __name__ == '__main__':
    pto_1_ant = (6, 0)
    pto_1_post = (12, 6)

    pto_2_ant = (12, 6)
    pto_2_post = (6, 12)

    pto_3_ant = (6, 12)
    pto_3_post = (0, 6)

    pto_4_ant = (0, 6)
    pto_4_post = (6, 0)

    bisector_1 = Perp_bisector(pto_1_ant, pto_1_post)
    bisector_2 = Perp_bisector(pto_2_ant, pto_2_post)
    bisector_3 = Perp_bisector(pto_3_ant, pto_3_post)
    bisector_4 = Perp_bisector(pto_4_ant, pto_4_post)

    i_12 = Intersection(bisector_1, bisector_2)
    i_14 = Intersection(bisector_1, bisector_4)
    i_23 = Intersection(bisector_2, bisector_3)
    i_34 = Intersection(bisector_3, bisector_4)

    int_x = [i_12.x, i_14.x, i_23.x, i_34.x]
    int_y = [i_12.y, i_14.y, i_23.y, i_34.y]
    print np.mean(int_x)
    print np.mean(int_y)
