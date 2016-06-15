#!/usr/bin/env python
import os
import csv
import cv2
import glob
import numpy as np

from measures.projection import Projection


class CoolRate_adv():
    def __init__(self):
        self.start = False
        self.laser_on = False
        self.p_NIT = Projection()
        self.p_NIT.load_configuration('../../config/NIT_config.yaml')

        # self.img = img
        # self.vel = vel

        self.time = None
        self.dt = None
        self.ds = None
        self.size = (500, 500, 3)
        self.frame_0 = np.zeros(self.size, dtype=np.uint8)

    def load_image(self, dir_frame):
        name = os.path.basename(dir_frame).split(".")[0]
        print name
        frame = cv2.imread(dir_frame)
        image = self.p_NIT.project_image(frame, self.p_NIT.hom_vis)
        return image

    def load_velocity(self, dir_file):
        with open(dir_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                print row[0]

    def load_data(self, dir_file, dir_frame):
        with open(dir_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            row = next(reader)
            for row in reader:
                t_v = row[0]
                # for f in sorted(dir_frame):
                #     print f

    def image_gradient(self, velocity, image):
        stamp = velocity.header.stamp
        vx = velocity.vx * 1000
        vy = velocity.vy * 1000
        vz = velocity.vz * 1000

        vel = np.float32([vx, vy, vz])

        self.get_ds(stamp.to_sec(), vel)
        gradient_image = np.zeros(self.size, dtype=np.uint8)

        for u in range(200, 300):
            for v in range(200, 300):
                pxl = np.float32([[u, v]])
                pxl_2 = self.p_NIT.transform(self.p_NIT.inv_hom_vis, pxl)
                pos = self.p_NIT.transform(self.p_NIT.inv_hom, pxl_2)
                gradient = self.get_gradient(vel, stamp, pos)
                if gradient is not None:
                    gradient_image[u, v] = self.convert_value(gradient)

        return gradient_image

    def get_ds(self, time, vel):
        if self.time is not None:
            dt = time - self.time
            self.ds = vel * dt
        self.time = time

    def get_gradient(self, vel, stamp, image, pos=np.float32([[0, 0]])):
        pos_0 = np.float32([[pos[0][0], pos[0][1], 0]])
        frame_1 = image
        pos_1 = self.get_position(pos_0)

        if pos_1 is not None:
            #get value of the pixel in:
                #frame_1: position_1
            pxl_pos_0 = self.p_NIT.transform(self.p_NIT.hom, pos_0)
            pxl_pos_0_vis = self.p_NIT.transform(self.p_NIT.hom_vis, pxl_pos_0)
            intensity_0 = self.get_value_pixel(self.frame_0, pxl_pos_0_vis[0])
                #frame_0: position_0
            pxl_pos_1 = self.p_NIT.transform(self.p_NIT.hom, pos_1)
            pxl_pos_1_vis = self.p_NIT.transform(self.p_NIT.hom_vis, pxl_pos_1)
            intensity_1 = self.get_value_pixel(frame_1, pxl_pos_1_vis[0])

            gradient = (intensity_1 - intensity_0)/self.coolrate.dt
            self.frame_0 = frame_1
            return gradient

        else:
            self.frame_0 = frame_1
            return None

    def get_position(self, position):
        if self.dt is not None:
            #position in mm
            position_1 = position + self.ds
            return position_1
        else:
            return None

    def get_value_pixel(self, frame, pxl, rng=3):
        intensity = 0
        limits = (rng - 1)/2
        for i in range(-limits, limits+1):
            for j in range(-limits, limits+1):
                index_i = pxl[0] + i
                index_j = pxl[1] + j
                intensity = intensity + frame[index_i, index_j]
        intensity = intensity/(rng*rng)
        return intensity

    def convert_value(self, gradient, inf_limit=-1200, sup_limit=1200):
        dp = 255.0/(sup_limit-inf_limit)
        grad = (gradient + 1200) * dp
        return grad

if __name__ == '__main__':
    coolrate = CoolRate_adv()
    print "velocity"
    vel_csv = "../../data/coolrate/topics/velocity/velocity.csv"
    vel = os.path.realpath(os.path.join(os.getcwd(), vel_csv))
    coolrate.load_velocity(vel)

    print "images"
    files_uEye = "../../data/coolrate/topics/camera/image/*.png"
    f_uEye = glob.glob(os.path.realpath(os.path.join(os.getcwd(), files_uEye)))
    coolrate.load_data(vel, f_uEye)
    # for f in sorted(f_uEye):
    #     image = coolrate.load_image(f)
