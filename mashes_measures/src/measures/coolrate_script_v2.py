#!/usr/bin/env python
import os
import csv
import cv2
import glob
import math
import numpy as np

from measures.projection import Projection
from matplotlib import pyplot as plt


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
        self.frame_0 = np.zeros((32, 32, 3), dtype=np.uint8)
        self.frame_1 = np.zeros((32, 32, 3), dtype=np.uint8)
        self.image = np.zeros((32, 32, 3), dtype=np.uint8)


        self.velocity = Velocity()

    def load_image(self, dir_frame):
        frame = cv2.imread(dir_frame, 1)
        # if frame.encoding == 'rgb8':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print frame.shape
        self.image = frame
        # if data.encoding == 'mono8':
        #     self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)

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

                frame_0 = self.frame_0.copy()
                image_resized_1 = cv2.resize(frame_0, (500, 500), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Pre", image_resized_1)

                image_gradient = self.image_gradient()
                # image_resized_2 = cv2.resize(image_gradient, (500, 500), interpolation=cv2.INTER_LINEAR)
                # plt.hist(image_resized_2.ravel(), 265, [0, 265])
                # plt.show()

                # cv2.imshow("image_gradient", image_resized_2)

                frame_1 = self.frame_1.copy()
                image_resized_3 = cv2.resize(frame_1, (500, 500), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Post", image_resized_3)

                dir_folder = os.path.abspath(os.path.join(os.getcwd(), "../../../mashes_calibration/data/coolrate/result"))
                if not os.path.exists(dir_folder):
                    os.makedirs(dir_folder)
                name_file = os.path.join(dir_folder, os.path.basename(d_f))

                cv2.imwrite(name_file, image_gradient)
                cv2.waitKey(1)

        print self.max_value, self.min_value
        max_v = np.array(self.max_value)
        min_v = np.array(self.min_value)
        print "Max, Min frame:", max_v.argmax(), min_v.argmin()
        print "Max:", np.amax(self.max_value), np.amax(self.min_value)
        print "Min:", np.amin(self.max_value), np.amin(self.min_value)

    def image_gradient(self):
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
        gradient_image = np.zeros((32, 32, 3), dtype=np.uint8)
        if self.dt is not None:
            pxl = [np.float32([[u, v]]) for u in range(0, 32) for v in range(0, 32)]
            pos = [self.p_NIT.transform(self.p_NIT.inv_hom, p) for p in pxl]

            x = y = np.arange(0, 255, 1)
            X, Y = np.meshgrid(x, y)

            gradient = np.array([self.get_gradient(vel, position) for position in pos])
            gradient = gradient.reshape((32, 32, 3))
            # print gradient
            self.max_value.append(np.nanmax(gradient))
            self.min_value.append(np.nanmin(gradient))

            # flat_grad = gradient.flatten()
            # plt.hist(flat_grad, bins=20)
            # plt.title("Histogram with 'auto' bins")
            # plt.show()
            gradient_image = np.array([self.convert_value(grad) for grad in gradient])
            gradient_image = gradient_image.reshape((32, 32, 3))
            print gradient_image

        self.frame_0 = self.frame_1
        return gradient_image

    def get_ds(self, time, vel):
        if self.time is not None:
            dt = time - self.time
            self.dt = dt
            self.ds = vel * dt
        self.time = time

    def get_gradient(self, vel, pos=np.float32([[0, 0]])):
        pos_0 = np.float32([[pos[0][0], pos[0][1], 0]])
        pos_1 = self.get_position(pos_0)
        if pos_1 is not None and self.dt < 0.2:
            #get value of the pixel in:
                #frame_1: position_1
            pxl_pos_0 = self.p_NIT.transform(self.p_NIT.hom, pos_0)
            intensity_0 = self.get_value_pixel(self.frame_0, pxl_pos_0[0])
        #         #frame_0: position_0
            pxl_pos_1 = self.p_NIT.transform(self.p_NIT.hom, pos_1)
            intensity_1 = self.get_value_pixel(self.frame_1, pxl_pos_1[0])

            if np.array_equal(intensity_0, np.float32([-1, -1, -1])) or np.array_equal(intensity_1, np.float32([-1, -1, -1])):
                gradient = np.float32([np.nan, np.nan, np.nan])
            else:
                gradient = (intensity_1 - intensity_0)/self.dt
            return gradient
        else:
            return np.float32([np.nan, np.nan, np.nan])

    def get_position(self, position):
        if self.dt is not None:
            position_1 = position + self.ds
            return position_1
        else:
            return None

    def get_value_pixel(self, frame, pxl, rng=3):
        intensity = np.float32([-1, -1, -1])
        limits = (rng - 1)/2
        if (pxl[0]-limits) < 0 or (pxl[0]+limits) > 31 or (pxl[1]-limits) < 0 or (pxl[1]+limits) > 31:
            return intensity
        else:
            intensity = np.float32([0, 0, 0])
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
