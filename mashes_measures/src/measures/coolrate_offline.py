#!/usr/bin/env python
import os
import csv
import cv2
import glob
import numpy as np

from measures.projection import Projection
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import stats

#-----------------------------additional classes------------------------------#


class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        # self.data = np.zeros(length, dtype='f')
        self.length = length
        self.data = [np.nan] * length
        self.index = 0
        self.complete = False
        self.stored_data = False

    def append_data(self, x):
        "adds array x to ring buffer"
        self.stored_data = True
        if self.index == self.length:
            self.complete = True

        if not self.complete:
            self.data[self.index] = x
            self.index = self.index + 1
        else:
            dt = self.data[1:]
            self.data[0:-1] = dt
            self.data[-1] = x

    def no_value(self):
        self.append_data(np.nan)


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
#-----------------------------additional classes------------------------------#


class CoolRate_adv():
    def __init__(self):
        #------------configurable variables------------#
        self.size_sensor = 32
        self.init_points = 2
        self.num_points = 10
        self.num_points_fitted = 4
        self.frame_sample = 48
        self.laser_threshold = 80
        self.rate_gradient = 4
        self.thr_no_laser = 40
        self.w_ppal = 0.9
        self.w_side = 0.015
        self.w_diag = 0.01
        self.scale_vis = 13
        self.pause_plot = 0.05
        self.pause_image = 100
        self.color = (255, 255, 255)

        self.points_plotted = 0
        self.num_plotted = 17
        self.font_size = 20
        self.line_width = 3
        #----------------------------------------------#

        self.start = False
        self.laser_on = False
        self.p_NIT = Projection()
        # self.p_NIT.load_configuration('../../../mashes_calibration/config/NIT_config.yaml')
        self.p_NIT.load_configuration('../../../mashes_calibration/config/tachyon.yaml')
        self.inv_hom = linalg.inv(self.p_NIT.hom)

        self.max_value = []
        self.max_i = []
        self.min_value = []
        self.min_i = []

        self.time = None
        self.dt = None
        self.ds = None

        self.first = True
        self.counter = 0

        self.frame_0 = np.zeros((self.size_sensor, self.size_sensor, 1), dtype=np.uint8)
        self.frame_1 = np.zeros((self.size_sensor, self.size_sensor, 1), dtype=np.uint8)
        self.image = np.zeros((self.size_sensor, self.size_sensor, 1), dtype=np.uint8)

        self.sizes = list(range(
            self.init_points + self.num_points, self.init_points, -1))
        self.matrix_intensity = [RingBuffer(a) for a in self.sizes]
        self.matrix_point = [RingBuffer(a) for a in self.sizes]
        self.matrix_pxl_point = [RingBuffer(a) for a in self.sizes]
        self.total_t = RingBuffer(self.num_points)
        self.dt_axis = []
        self.t_axis = []
        self.total_intensity = []

        self.coeff_1 = []
        self.coeff_2 = []
        self.coeff_3 = []

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

        axisNum = 0
        with open(dir_file, 'rb') as csvfile:

            fig_0 = plt.figure()
            # fig_0.suptitle('Data and Fitted Function')
            ax1_0 = fig_0.add_subplot(111)
            # ax1_0.set_title('Data')
            ax1_0.set_xlabel('ms', fontsize=self.font_size)
            ax1_0.set_ylabel('counts', fontsize=self.font_size)
            ax1_0.set_xlim([0, self.frame_sample*(self.num_points-1)])
            ax1_0.set_ylim([0, 256])
            ax1_0.tick_params(labelsize=self.font_size)


            reader = csv.reader(csvfile, delimiter=',')
            row = next(reader)
            i_prev = 0
            i_frame = 0
            for row in reader:
                i_prev = i_frame
                t_v = int(row[0])
                matrix = [abs(x - t_v) for x in t_frames]
                i_frame = matrix.index(min(matrix))

                if i_frame > i_prev:
                    for i in range(i_prev, i_frame):
                        self.counter = self.counter + 1
                        if self.counter == self.frame_sample:
                            self.counter = 0
                            t_f = t_frames[i]
                            d_f = d_frames[i]
                            print "tv-", t_v, "tf-", t_f, "index-", i, "name file-", d_f
                            self.t = t_f
                            self.load_image(d_f)
                            self.velocity.load_velocity(row)

                            x = y = np.arange(0, self.size_sensor, 1)
                            X, Y = np.meshgrid(x, y)
                            index = self.get_maxvalue()

                            if index is not None:
                                self.image_gradient(index)
                            else:
                                if self.matrix_intensity[0].stored_data:
                                    self.image_gradient(index, False)


                            if self.matrix_intensity[0].index == self.matrix_intensity[0].length:
                                intensity = [self.matrix_intensity[point].data[0] for point in range(self.num_points)]

                                #vvvvviiiiiissssssssssssssssssssssssssssssssss#
                                img = self.image.copy()

                                self.total_intensity.append(intensity)
                                x = np.arange(self.num_points).tolist()
                                x_a = np.array(x)
                                intensity_f = np.array(intensity[:self.num_points_fitted])
                                x_f = np.array(np.arange(self.num_points_fitted).tolist())
                                # Exponential Fit (Note that we have to provide the y-offset ("C") value!!
                                A, K = self.fit_exp_linear(x_f, intensity_f, self.thr_no_laser)
                                fit_y = self.model_func(x_a, A, K, self.thr_no_laser)
                                self.coeff_1.append(A)
                                self.coeff_2.append(K)

                                colors = ('b', 'g',  'y', 'r', 'm', 'c', 'k')
                                set_time = [x*self.frame_sample for x in range(self.num_points)]

                                if self.points_plotted == self.num_plotted:
                                    axisNum += 1
                                    color = colors[axisNum % len(colors)]
                                    plt.ion()
                                    ax1_0.plot(set_time, intensity, color=color, linewidth=self.line_width)
                                    plt.pause(self.pause_plot)


                                    plt.ion()
                                    ax1_0.plot(set_time, fit_y, '--', color=color, linewidth=self.line_width)
                                    plt.pause(self.pause_plot)
                                    self.points_plotted = 0
                                else:
                                    self.points_plotted = self.points_plotted + 1


                                w, h = img.shape
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                                img_rgb = cv2.applyColorMap(img_rgb, cv2.COLORMAP_JET)
                                img_plus = cv2.resize(img_rgb, (w*self.scale_vis, h*self.scale_vis), interpolation=cv2.INTER_LINEAR)
                                pxl = [self.matrix_pxl_point[t].data[0] for t in range(self.num_points)]
                                for p in pxl:
                                    if not np.isnan(p).any():
                                        cv2.circle(img_plus, (int(round(p[0][0])*self.scale_vis), int(round(p[0][1])*self.scale_vis)), 4, self.color, -1)
                                cv2.imshow("Image: ", img_plus)
                                cv2.waitKey(self.pause_image)
                                self.t_axis.append(float(self.total_t.data[0])/1000000)
                                print "Self.dt", self.dt
                                #vvvvviiiiiissssssssssssssssssssssssssssssssss#


            x_t = [self.t_axis[t] - self.t_axis[0] for t in range(len(self.t_axis))]

            print self.coeff_2
            print x_t
            fig_1 = plt.figure()
            fig_1.suptitle('Fitted Function:\n y = A e^(K t) + 0')

            ax1 = fig_1.add_subplot(211)
            ax1.plot(x_t, self.coeff_1, linewidth=self.line_width)
            ax1.set_title('A', fontsize=self.font_size)
            ax1.set_xlabel('ms', fontsize=self.font_size)
            ax1.set_ylabel('value', fontsize=self.font_size)
            ax1.tick_params(labelsize=self.font_size)
            ax1.set_ylim([0, 400])
            ax2 = fig_1.add_subplot(212)
            ax2.plot(x_t, self.coeff_2, linewidth=self.line_width)
            ax2.set_title('K', fontsize=self.font_size)
            ax2.set_xlabel('ms', fontsize=self.font_size)
            ax2.set_ylabel('value', fontsize=self.font_size)
            ax2.tick_params(labelsize=self.font_size)
            ax2.set_ylim([-1, 0])


            mean_coeff_1 = np.nanmean(self.coeff_1)
            mean_coeff_2 = np.nanmean(self.coeff_2)

            print "Media aritmetica:\n A:", mean_coeff_1, "\n K:", mean_coeff_2

            median_coeff_1 = np.nanmedian(self.coeff_1)
            median_coeff_2 = np.nanmedian(self.coeff_2)

            print "Mediana ( valor central datos):\n A:", median_coeff_1, "\n K:", median_coeff_2

            var_coeff_1 = np.nanvar(self.coeff_1)
            var_coeff_2 = np.nanvar(self.coeff_2)

            print "Varianza (dispersion de los datos):\n A:", var_coeff_1, "\n K:", var_coeff_2

            std_coeff_1 = np.nanstd(self.coeff_1)
            std_coeff_2 = np.nanstd(self.coeff_2)

            print "Desv. tipica (raiz cuadrada desv. tipica):\n A:", std_coeff_1, "\n K:", std_coeff_2

            mode_coeff_1 = stats.mode(self.coeff_1)
            mode_coeff_2 = stats.mode(self.coeff_2)

            print "Moda (Valor con mayor frecuencia abs):\n A:", mode_coeff_1, "\n K:", mode_coeff_2

            plt.show()

            while True:
                plt.pause(self.pause_plot)

    def fit_exp_linear(self, t, y, C=0):
        y = y - C
        y = np.log(y)
        K, A_log = np.polyfit(t, y, 1)
        A = np.exp(A_log)
        return A, K

    def model_func(self, t, A, K, C):
        return A * np.exp(K * t) + C

    def get_maxvalue(self, rng=3):
        image = np.zeros((self.size_sensor, self.size_sensor), dtype=np.uint16)
        pxls = [np.float32([u, v]) for u in range(0, self.size_sensor) for v in range(0, self.size_sensor)]
        for pxl in pxls:
            index = pxl[0], pxl[1]
            intensity = self.get_value_pixel(self.image, pxl)
            if intensity == -1:
                image[index] = 0
            else:
                image[index] = intensity

        value = np.amax(image)
        i = np.unravel_index(np.argmax(image), image.shape)
        print value, i
        if value > self.laser_threshold:
            print "laser on"
            return i
        else:
            return None

    def get_value_pixel(self, frame, pxl, rng=3):
        intensity = -1
        limits = (rng - 1)/2
        p_0 = round(pxl[0])
        p_1 = round(pxl[1])
        if (p_0-limits) < 0 or (p_0+limits) > (self.size_sensor-1) or (p_1-limits) < 0 or (p_1+limits) > (self.size_sensor-1):
            return intensity
        else:
            intensity = 0
            if rng == 3:
                for i in range(-limits, limits+1):
                    for j in range(-limits, limits+1):
                        index_i = pxl[0] + i
                        index_j = pxl[1] + j
                        if i == 0 and j == 0:
                            intensity = intensity + self.image[index_i, index_j]*self.w_ppal
                        elif i == 0 or j == 0:
                            intensity = intensity + self.image[index_i, index_j]*self.w_side
                        else:
                            intensity = intensity + self.image[index_i, index_j]*self.w_diag
            else:
                for i in range(-limits, limits+1):
                    for j in range(-limits, limits+1):
                        index_i = p_0 + i
                        index_j = p_1 + j
                        intensity = intensity + frame[index_i, index_j]
                intensity = intensity/(rng*rng)
            return intensity

    def image_gradient(self, pxl_pos, data=True):
        if self.first:
            if data:
                self.first = False
                #mm/s
                vx = self.velocity.vx*1000
                vy = self.velocity.vy*1000
                vz = self.velocity.vz*1000
                vel = np.float32([vx, vy, vz])
                self.get_ds(self.t, vel)

                self.frame_0 = self.image

                pxl_pos_0 = np.float32([[pxl_pos[0], pxl_pos[1]]])
                pos_0 = self.p_NIT.transform(self.inv_hom, pxl_pos_0)
                intensity_0 = self.get_value_pixel(self.frame_0, pxl_pos_0[0])

                self.matrix_intensity[0].append_data(intensity_0)
                self.matrix_pxl_point[0].append_data(pxl_pos_0)
                self.matrix_point[0].append_data(pos_0)
        else:
            #mm/s
            vx = self.velocity.vx*1000
            vy = self.velocity.vy*1000
            vz = self.velocity.vz*1000
            vel = np.float32([vx, vy, vz])
            self.get_ds(self.t, vel)

            self.frame_1 = self.frame_0
            self.frame_0 = self.image

            if data:
                pxl_pos_0 = np.float32([[pxl_pos[0], pxl_pos[1]]])
                pos_0 = self.p_NIT.transform(self.inv_hom, pxl_pos_0)
                intensity_0 = self.get_value_pixel(self.frame_0, pxl_pos_0[0])
            else:
                pxl_pos_0 = np.nan
                pos_0 = np.nan
                intensity_0 = np.nan

            self.matrix_intensity[0].append_data(intensity_0)
            self.matrix_pxl_point[0].append_data(pxl_pos_0)
            self.matrix_point[0].append_data(pos_0)

            if not self.matrix_intensity[0].complete:
                last_index = self.matrix_point[0].index - 2
                if last_index < self.num_points-1:
                    for x, y in zip(range(last_index, -1, -1), range(last_index+1)):
                        self.get_next_data(x, y)
                else:
                    ly_1 = self.matrix_intensity[self.num_points-2].index - 1
                    for x, y in zip(range(self.num_points-2, -1, -1), range(ly_1, last_index+1)):
                        self.get_next_data(x, y)
            else:
                lx = len(self.matrix_intensity) - 2
                ly_1 = len(self.matrix_intensity[0].data) - 2
                ly_2 = len(self.matrix_intensity[-2].data) - 2
                for x, y in zip(range(lx + 1), range(ly_1, ly_2 - 1, -1)):
                    self.get_next_data(x, y)

    def get_next_data(self, x, y):
        intensity_0 = self.matrix_intensity[x].data[y]
        pos_0 = np.float32(self.matrix_point[x].data[y])
        pxl_pos_0 = np.float32(self.matrix_pxl_point[x].data[y])
        if not np.isnan(intensity_0):
            pos_1, pxl_pos_1, intensity_1 = self.get_next_value(vel, intensity_0, pxl_pos_0, pos_0)
            if pos_1 is not None and pxl_pos_1 is not None and intensity_1 is not None:
                pxl = np.float32([[pxl_pos_1[0][0], pxl_pos_1[0][1]]])
                self.matrix_intensity[x+1].append_data(intensity_1)
                self.matrix_pxl_point[x+1].append_data(pxl)
                self.matrix_point[x+1].append_data(pos_1)
        else:
                self.matrix_intensity[x+1].append_data(np.nan)
                self.matrix_pxl_point[x+1].append_data(np.nan)
                self.matrix_point[x+1].append_data(np.nan)

    def get_ds(self, time, vel):
        if self.time is not None:
            dt = time - self.time
            self.dt = dt
            self.total_t.append_data(self.time)
            self.ds = vel * dt / 1000000000
        self.time = time
        print "timeeeee", self.dt

    def get_gradient(self, rate):
        intensity_0 = self.matrix_intensity[0].data[0]
        intensity_1 = self.matrix_intensity[rate].data[0]
        if intensity_0 == -1 or intensity_1 == -1:
            gradient = np.nan
        else:
            gradient = intensity_1 - intensity_0

    def get_next_value(self, vel, intensity_0, pxl_pos_0, pos=np.float32([[0, 0]])):
        pos_0 = np.float32([[pos[0][0], pos[0][1], 0]])
        pos_1 = self.get_position(pos_0)
        if pos_1 is not None and self.dt < 0.2 * 1000000000:
        #         #frame_0: position_0
            pxl_pos_1 = self.p_NIT.transform(self.p_NIT.hom, pos_1)
            intensity_1 = self.get_value_pixel(self.frame_1, pxl_pos_1[0])

            # if intensity_0 == -1 or intensity_1 == -1:
            #     gradient = np.nan
            # else:
            #     gradient = (intensity_1 - intensity_0)/self.dt

            return pos_1, pxl_pos_1, intensity_1
        else:
            return None, None, None

    def get_position(self, position):
        if self.dt is not None:
            position_1 = position + self.ds
            return position_1
        else:
            return None

if __name__ == '__main__':
    coolrate = CoolRate_adv()

    vel_csv = "../../../mashes_calibration/data/coolrate/velocity/velocity.csv"
    vel = os.path.realpath(os.path.join(os.getcwd(), vel_csv))

    files_NIT = "../../../mashes_calibration/data/coolrate/tachyon/image/*.png"
    f_NIT = glob.glob(os.path.realpath(os.path.join(os.getcwd(), files_NIT)))
    coolrate.load_data(vel, f_NIT)
