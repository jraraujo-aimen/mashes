#!/usr/bin/env python
import os
import csv
import cv2
import glob
import time
import numpy as np
from collections import deque

from measures.projection_Noemi import Projection
import matplotlib.pyplot as plt
import matplotlib as mpl


class Velocity():
    def __init__(self):
        self.t_v = 0
        self.speed = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0


    def load_velocity(self, row):
        self.t_v = float(row[0])
        self.speed = float(row[1])
        self.vx = float(row[2])
        self.vy = float(row[3])
        self.vz = float(row[4])


class CoolRate_adv():
    def __init__(self):
        self.color = (0, 0, 0)
        self.start = False
        self.laser_on = False
        self.p_NIT = Projection()
        self.p_NIT.load_configuration('../../../mashes_calibration/config/NIT_config.yaml')

        self.max_value = []
        self.max_i = []
        self.min_value = []
        self.min_i = []

        self.time = None
        self.dt = None
        self.ds = None
        self.size = (500, 500, 3)
        self.frame_0 = np.zeros((32, 32, 1), dtype=np.uint8)
        self.frame_1 = np.zeros((32, 32, 1), dtype=np.uint8)
        self.image = np.zeros((32, 32, 1), dtype=np.uint8)
        self.buffer_1 = deque(maxlen=16)
        self.buffer_2 = deque(maxlen=15)
        self.buffer_3 = deque(maxlen=14)
        self.buffer_4 = deque(maxlen=13)
        self.buffer_5 = deque(maxlen=12)
        self.buffer_6 = deque(maxlen=11)
        self.buffer_7 = deque(maxlen=10)
        self.buffer_8 = deque(maxlen=9)
        self.buffer_9 = deque(maxlen=8)

        self.gradient_1 = deque(maxlen=16)
        self.gradient_2 = deque(maxlen=15)
        self.gradient_3 = deque(maxlen=14)
        self.gradient_4 = deque(maxlen=13)
        self.gradient_5 = deque(maxlen=12)
        self.gradient_6 = deque(maxlen=11)
        self.gradient_7 = deque(maxlen=10)
        self.gradient_8 = deque(maxlen=9)
        self.gradient_9 = deque(maxlen=8)

        self.intensity_1 = deque(maxlen=16)
        self.intensity_2 = deque(maxlen=15)
        self.intensity_3 = deque(maxlen=14)
        self.intensity_4 = deque(maxlen=13)
        self.intensity_5 = deque(maxlen=12)
        self.intensity_6 = deque(maxlen=11)
        self.intensity_7 = deque(maxlen=10)
        self.intensity_8 = deque(maxlen=9)
        self.intensity_9 = deque(maxlen=8)

        self.total_gradient = []
        self.total_intensity = []

        self.velocity = Velocity()

    def load_image(self, dir_frame):
        frame = cv2.imread(dir_frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.image = frame

    def load_data(self, dir_file, dir_frame):
        t_frames = []
        d_frames = []
        for f in sorted(dir_frame):
            d_frames.append(f)
            t_frame = int(os.path.basename(f).split(".")[0])
            t_frames.append(t_frame)

        with open(dir_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            row = next(reader)
            for row in reader:
                t_v = int(row[0])
                matrix = [abs(x - t_v) for x in t_frames]
                i_frame = matrix.index(min(matrix))
                t_f = t_frames[i_frame]
                d_f = d_frames[i_frame]
                print "tiempos:", t_v, t_f, i_frame, d_f
                self.load_image(d_f)
                self.velocity.load_velocity(row)

                x = y = np.arange(0, 32, 1)
                X, Y = np.meshgrid(x, y)
                index = self.get_maxvalue()
                if index is not None:
                    self.image_gradient(index)

                    # if self.buffer_9:
                    #     if len(self.buffer_9) == 8:
                    #         gradient_array = [self.gradient_2[0], self.gradient_3[0], self.gradient_4[0], self.gradient_5[0], self.gradient_6[0], self.gradient_7[0], self.gradient_8[0], self.gradient_9[0]]
                    #         self.total_gradient.append(gradient_array)
                    #         # print "Gradient profile:", gradient_array

                    if self.intensity_9:
                        if len(self.intensity_9) == 8:
                            intensity_array = [self.intensity_1[0], self.intensity_2[0], self.intensity_3[0], self.intensity_4[0], self.intensity_5[0], self.intensity_6[0], self.intensity_7[0], self.intensity_8[0], self.intensity_9[0]]
                            self.total_intensity.append(intensity_array)
                            # print "Gradient profile:", gradient_array

            # print "Total Gradient profile:", self.total_gradient
            # num_g = [1, 2, 3, 4, 5, 6, 7, 8]
            # total_gradient = np.array(self.total_gradient)
            # n_g, _ = total_gradient.shape
            # colors_g = mpl.cm.rainbow(np.linspace(0, 1, n_g))
            # for color, y in zip(colors_g, total_gradient):
            #     plt.plot(num_g, y, color=color)
            # plt.show()


            print "Total Intensity profile:", self.total_intensity
            num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            total_intensity = np.array(self.total_intensity)
            n_i, _ = total_intensity.shape
            colors = mpl.cm.rainbow(np.linspace(0, 1, n_i))
            for color, y in zip(colors, total_intensity):
                plt.plot(num, y, color=color)
            plt.show()

    def get_maxvalue(self, rng=3):
        image = np.zeros((32, 32), dtype=np.uint16)
        pxls = [np.float32([u, v]) for u in range(0, 32) for v in range(0, 32)]
        for pxl in pxls:
            value = 0
            index = pxl[0], pxl[1]
            limits = (rng - 1)/2
            if (pxl[0]-limits) < 0 or (pxl[0]+limits) > 31 or (pxl[1]-limits) < 0 or (pxl[1]+limits) > 31:
                image[index] = 0
            else:
                for i in range(-limits, limits+1):
                    for j in range(-limits, limits+1):
                        index_i = pxl[0] + i
                        index_j = pxl[1] + j
                        value = value + self.image[index_i, index_j]
                value = value/(rng*rng)
                image[index] = value

        value = np.amax(image)
        i = np.unravel_index(np.argmax(image), image.shape)
        print value, i
        if value > 150:
            print "laser on"
            return i
        else:
            return None
        time.sleep(1)

    def image_gradient(self, pxl_pos):
        #ns
        t = self.velocity.t_v/1000000000
        #mm/s
        vx = self.velocity.vx*1000
        vy = self.velocity.vy*1000
        vz = self.velocity.vz*1000
        vel = np.float32([vx, vy, vz])

        self.get_ds(t, vel)

        self.frame_1 = self.image

        print "image_gradient"
        if pxl_pos is not None:

            if self.buffer_8:
                pxl_pos_0 = self.buffer_8[-1]
                pos = self.p_NIT.transform(self.p_NIT.inv_hom, pxl_pos_0)
                self.get_gradient(vel, pxl_pos_0, pos, 7)

            if self.buffer_7:
                pxl_pos_0 = self.buffer_7[-1]
                pos = self.p_NIT.transform(self.p_NIT.inv_hom, pxl_pos_0)
                self.get_gradient(vel, pxl_pos_0, pos, 6)

            if self.buffer_6:
                pxl_pos_0 = self.buffer_6[-1]
                pos = self.p_NIT.transform(self.p_NIT.inv_hom, pxl_pos_0)
                self.get_gradient(vel, pxl_pos_0, pos, 5)

            if self.buffer_5:
                pxl_pos_0 = self.buffer_5[-1]
                pos = self.p_NIT.transform(self.p_NIT.inv_hom, pxl_pos_0)
                self.get_gradient(vel, pxl_pos_0, pos, 4)

            if self.buffer_4:
                pxl_pos_0 = self.buffer_4[-1]
                pos = self.p_NIT.transform(self.p_NIT.inv_hom, pxl_pos_0)
                self.get_gradient(vel, pxl_pos_0, pos, 3)

            if self.buffer_3:
                pxl_pos_0 = self.buffer_3[-1]
                pos = self.p_NIT.transform(self.p_NIT.inv_hom, pxl_pos_0)
                self.get_gradient(vel, pxl_pos_0, pos, 2)

            if self.buffer_2:
                pxl_pos_0 = self.buffer_2[-1]
                pos = self.p_NIT.transform(self.p_NIT.inv_hom, pxl_pos_0)
                self.get_gradient(vel, pxl_pos_0, pos, 1)

            pxl_pos_0 = np.float32([[pxl_pos[0], pxl_pos[1]]])
            pos = self.p_NIT.transform(self.p_NIT.inv_hom, pxl_pos_0)
            self.buffer_1.append(pxl_pos_0)
            self.get_gradient(vel, pxl_pos_0, pos)


            # print "2: ", self.gradient_2
            # print "3: ", self.gradient_3
            # print "4: ", self.gradient_4
            # print "5: ", self.gradient_5
            # print "6: ", self.gradient_6
            # print "7: ", self.gradient_7


        self.frame_0 = self.frame_1
        # return gradient

    def get_ds(self, time, vel):
        if self.time is not None:
            dt = time - self.time
            self.dt = dt
            self.ds = vel * dt
        self.time = time

    def get_gradient(self, vel, pxl_pos_0, pos=np.float32([[0, 0]]), num_iter=0):
        pos_0 = np.float32([[pos[0][0], pos[0][1], 0]])
        pos_1 = self.get_position(pos_0)
        if pos_1 is not None and self.dt < 0.2:
            #get value of the pixel in:
                #frame_1: position_1
            intensity_0 = self.get_value_pixel(self.frame_0, pxl_pos_0[0])
        #         #frame_0: position_0
            pxl_pos_1 = self.p_NIT.transform(self.p_NIT.hom, pos_1)

            intensity_1 = self.get_value_pixel(self.frame_1, pxl_pos_1[0])

            # print intensity_0, intensity_1

            if intensity_0 == -1 or intensity_1 == -1:
                gradient = np.nan
            else:
                gradient = (intensity_1 - intensity_0)/self.dt

            if intensity_0 == -1:
                intensity_0 = np.nan
            if intensity_1 == -1:
                intensity_1 = np.nan

            if num_iter == 0:
                self.buffer_2.append(pxl_pos_1)
                self.gradient_2.append(gradient)
                self.intensity_1.append(intensity_0)
                self.intensity_2.append(intensity_1)
            elif num_iter == 1:
                self.buffer_3.append(pxl_pos_1)
                self.gradient_3.append(gradient)
                self.intensity_3.append(intensity_1)
            elif num_iter == 2:
                self.buffer_4.append(pxl_pos_1)
                self.gradient_4.append(gradient)
                self.intensity_4.append(intensity_1)
            elif num_iter == 3:
                self.buffer_5.append(pxl_pos_1)
                self.gradient_5.append(gradient)
                self.intensity_5.append(intensity_1)
            elif num_iter == 4:
                self.buffer_6.append(pxl_pos_1)
                self.gradient_6.append(gradient)
                self.intensity_6.append(intensity_1)
            elif num_iter == 5:
                self.buffer_7.append(pxl_pos_1)
                self.gradient_7.append(gradient)
                self.intensity_7.append(intensity_1)
            elif num_iter == 6:
                self.buffer_8.append(pxl_pos_1)
                self.gradient_8.append(gradient)
                self.intensity_8.append(intensity_1)
            elif num_iter == 7:
                self.buffer_9.append(pxl_pos_1)
                self.gradient_9.append(gradient)
                self.intensity_9.append(intensity_1)


    def get_position(self, position):
        if self.dt is not None:
            position_1 = position + self.ds
            return position_1
        else:
            return None

    def get_value_pixel(self, frame, pxl, rng=3):
        intensity = -1
        limits = (rng - 1)/2
        if (pxl[0]-limits) < 0 or (pxl[0]+limits) > 31 or (pxl[1]-limits) < 0 or (pxl[1]+limits) > 31:
            return intensity
        else:
            intensity = 0
            for i in range(-limits, limits+1):
                for j in range(-limits, limits+1):
                    index_i = pxl[0] + i
                    index_j = pxl[1] + j
                    intensity = intensity + frame[index_i, index_j]
            intensity = intensity/(rng*rng)
            return intensity

    def convert_value(self, gradient, inf_limit=-2000, sup_limit=2000):
        if np.array_equal(gradient, np.float32([np.nan, np.nan, np.nan])):
            grad = np.float32([np.nan, np.nan, np.nan])
        else:
            dp = 255.0/(sup_limit-inf_limit)
            grad = (gradient - inf_limit) * dp
        return grad


if __name__ == '__main__':
    coolrate = CoolRate_adv()

    vel_csv = "../../../mashes_calibration/data/coolrate/velocity/velocity.csv"
    vel = os.path.realpath(os.path.join(os.getcwd(), vel_csv))

    files_NIT = "../../../mashes_calibration/data/coolrate/tachyon/image/*.png"
    f_NIT = glob.glob(os.path.realpath(os.path.join(os.getcwd(), files_NIT)))
    coolrate.load_data(vel, f_NIT)
