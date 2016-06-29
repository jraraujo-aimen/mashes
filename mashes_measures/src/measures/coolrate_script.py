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
        self.p_NIT.load_configuration('../../../mashes_calibration/config/NIT_config.yaml')

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
<<<<<<< HEAD
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
                image_resized_2 = cv2.resize(image_gradient, (500, 500), interpolation=cv2.INTER_LINEAR)
                # plt.hist(image_resized_2.ravel(), 265, [0, 265])
                # plt.show()

                cv2.imshow("image_gradient", image_resized_2)

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
=======
                t_v = row[0]
                # for f in sorted(dir_frame):
                #     print f

    def image_gradient(self, velocity, image):
        stamp = velocity.header.stamp
        vx = velocity.vx * 1000
        vy = velocity.vy * 1000
        vz = velocity.vz * 1000

>>>>>>> parent of 290dcf2... Update: mashes_calibration code
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
<<<<<<< HEAD

    vel_csv = "../../../mashes_calibration/data/coolrate/velocity/velocity.csv"
    vel = os.path.realpath(os.path.join(os.getcwd(), vel_csv))

    files_NIT = "../../../mashes_calibration/data/coolrate/tachyon/image/*.png"
    f_NIT = glob.glob(os.path.realpath(os.path.join(os.getcwd(), files_NIT)))
    coolrate.load_data(vel, f_NIT)
=======
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
>>>>>>> parent of 290dcf2... Update: mashes_calibration code
