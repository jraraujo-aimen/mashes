import os
import csv
import cv2
import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

from measures.projection import Projection

WHITE = (255, 255, 255)
FONT_SIZE = 20
LINE_WIDTH = 3
SCALE_VIS = 13
PAUSE_PLOT = 0.05
PAUSE_IMAGE = 100
SIZE_SENSOR = 32
NUM_PLOTTED = 17
NUM_POINTS = 10
INIT_POINTS = 2
NUM_POINTS_FITTED = 4
COLORS = ('b', 'g',  'y', 'r', 'm', 'c', 'k')

FRAME_SAMPLE = 48
W_PPAL = 0.9
W_SIDE = 0.015
W_DIAG = 0.01


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


class CoolRate():
    def __init__(self):
        self.laser_threshold = 80
        self.thr_no_laser = 40
        self.points_plotted = 0
        #----------------------------------------------#

        self.p_NIT = Projection()
        self.p_NIT.load_configuration('../../config/tachyon.yaml')

        self.time = None
        self.dt = None
        self.ds = None

        self.first = True

        self.frame_1 = np.zeros((SIZE_SENSOR, SIZE_SENSOR, 1), dtype=np.uint8)

        self.sizes = list(range(
            INIT_POINTS + NUM_POINTS, INIT_POINTS, -1))
        self.matrix_intensity = [RingBuffer(a) for a in self.sizes]
        self.matrix_point = [RingBuffer(a) for a in self.sizes]
        self.matrix_pxl_point = [RingBuffer(a) for a in self.sizes]
        self.total_t = RingBuffer(NUM_POINTS)
        self.dt_axis = []
        self.t_axis = []

        self.coeff_1 = []
        self.coeff_2 = []

    def load_image(self, dir_frame):
        frame = cv2.imread(dir_frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def load_frames(self, dir_frame):
        t_frames, d_frames = [], []
        for f in sorted(dir_frame):
            d_frames.append(self.load_image(f))
            t_frame = int(os.path.basename(f).split(".")[0])
            t_frames.append(t_frame)
        return t_frames, d_frames

    def get_velocity(self, row):
        t_v = int(row[0])
        speed = float(row[1])
        vx = float(row[2]) * 1000
        vy = float(row[3]) * 1000
        vz = float(row[4]) * 1000
        vel = np.float32([vx, vy, vz])
        return t_v, vel

    def load_velocities(self, dir_file):
        velocities = []
        with open(dir_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            row = next(reader)
            for row in reader:
                velocities(self.get_velocity(row))
        return velocities

    def load_data(self, dir_file, dir_frame):
        velocities = self.load_velocities(dir_file)
        t_frames, d_frames = self.load_frames(dir_frame)

        fig_0 = plt.figure()
        # fig_0.suptitle('Data and Fitted Function')
        ax1_0 = fig_0.add_subplot(111)
        # ax1_0.set_title('Data')
        ax1_0.set_xlabel('ms', fontsize=FONT_SIZE)
        ax1_0.set_ylabel('counts', fontsize=FONT_SIZE)
        ax1_0.set_xlim([0, FRAME_SAMPLE*(NUM_POINTS-1)])
        ax1_0.set_ylim([0, 256])
        ax1_0.tick_params(labelsize=FONT_SIZE)

        counter = 0
        axisNum = 0
        i_prev = 0
        i_frame = 0
        for t, vel in velocities:
            ds = self.get_ds(t, vel)
            i_prev = i_frame
            i_frame = np.argmin(np.array([abs(x - t) for x in t_frames]))
            if i_frame > i_prev:
                for i in range(i_prev, i_frame):
                    counter = counter + 1
                    if counter == FRAME_SAMPLE:
                        counter = 0
                        t = t_frames[i]
                        image = d_frames[i]
                        index = self.get_maxvalue(image)
                        if index is not None:
                            self.image_gradient(image, ds, index)
                        else:
                            if self.matrix_intensity[0].stored_data:
                                self.image_gradient(image, ds, index, False)

                        if self.matrix_intensity[0].index == self.matrix_intensity[0].length:
                            intensity = [self.matrix_intensity[point].data[0] for point in range(NUM_POINTS)]

                            #vvvvviiiiiissssssssssssssssssssssssssssssssss#
                            x_a = np.arange(NUM_POINTS)
                            x_f = np.arange(NUM_POINTS_FITTED)
                            intensity_f = np.array(intensity[:NUM_POINTS_FITTED])

                            # Exponential Fit (Note that we have to provide the y-offset ("C") value!!
                            A, K = self.fit_exp_linear(x_f, intensity_f, self.thr_no_laser)
                            fit_y = self.model_func(x_a, A, K, self.thr_no_laser)
                            self.coeff_1.append(A)
                            self.coeff_2.append(K)

                            set_time = [x*FRAME_SAMPLE for x in range(NUM_POINTS)]
                            if self.points_plotted == NUM_PLOTTED:
                                axisNum += 1

                                color = COLORS[axisNum % len(COLORS)]
                                plt.ion()
                                ax1_0.plot(set_time, intensity, color=color, linewidth=LINE_WIDTH)
                                plt.pause(PAUSE_PLOT)
                                plt.ion()
                                ax1_0.plot(set_time, fit_y, '--', color=color, linewidth=LINE_WIDTH)
                                plt.pause(PAUSE_PLOT)

                                self.points_plotted = 0
                            else:
                                self.points_plotted = self.points_plotted + 1

                            self.show_image(image.copy())

                            self.t_axis.append(float(self.total_t.data[0])/1000000)
                            print "Self.dt", self.dt
                            #vvvvviiiiiissssssssssssssssssssssssssssssssss#

        x_t = [self.t_axis[t] - self.t_axis[0] for t in range(len(self.t_axis))]

        self.plot_fitted_data(x_t)
        self.print_stats()

    def fit_exp_linear(self, t, y, C=0):
        y = y - C
        y = np.log(y)
        K, A_log = np.polyfit(t, y, 1)
        A = np.exp(A_log)
        return A, K

    def model_func(self, t, A, K, C):
        return A * np.exp(K * t) + C

    def show_image(self, img):
        w, h = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_rgb = cv2.applyColorMap(img_rgb, cv2.COLORMAP_JET)
        img_plus = cv2.resize(img_rgb, (w*SCALE_VIS, h*SCALE_VIS), interpolation=cv2.INTER_LINEAR)
        pxl = [self.matrix_pxl_point[t].data[0] for t in range(NUM_POINTS)]
        for p in pxl:
            if not np.isnan(p).any():
                cv2.circle(img_plus, (int(round(p[0][0])*SCALE_VIS), int(round(p[0][1])*SCALE_VIS)), 4, WHITE, -1)
        cv2.imshow("Image: ", img_plus)
        cv2.waitKey(PAUSE_IMAGE)

    def plot_fitted_data(self, x_t):
        print self.coeff_1, self.coeff_2, x_t
        fig_1 = plt.figure()
        fig_1.suptitle('Fitted Function:\n y = A e^(K t) + 0')
        ax1 = fig_1.add_subplot(211)
        ax1.plot(x_t, self.coeff_1, linewidth=LINE_WIDTH)
        ax1.set_title('A', fontsize=FONT_SIZE)
        ax1.set_xlabel('ms', fontsize=FONT_SIZE)
        ax1.set_ylabel('value', fontsize=FONT_SIZE)
        ax1.tick_params(labelsize=FONT_SIZE)
        ax1.set_ylim([0, 400])
        ax2 = fig_1.add_subplot(212)
        ax2.plot(x_t, self.coeff_2, linewidth=LINE_WIDTH)
        ax2.set_title('K', fontsize=FONT_SIZE)
        ax2.set_xlabel('ms', fontsize=FONT_SIZE)
        ax2.set_ylabel('value', fontsize=FONT_SIZE)
        ax2.tick_params(labelsize=FONT_SIZE)
        ax2.set_ylim([-1, 0])
        plt.show()

    def print_stats(self):
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

    def get_maxvalue(self, image, rng=3):
        image = np.zeros((SIZE_SENSOR, SIZE_SENSOR), dtype=np.uint16)
        pxls = [np.float32([u, v]) for u in range(0, SIZE_SENSOR) for v in range(0, SIZE_SENSOR)]
        for pxl in pxls:
            index = pxl[0], pxl[1]
            intensity = self.get_value_pixel(image, pxl)
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

    def get_value_pixel(self, image, pxl, rng=3):
        intensity = -1
        limits = (rng - 1)/2
        p_0, p_1 = round(pxl[0]), round(pxl[1])
        if (p_0-limits) < 0 or (p_0+limits) > (SIZE_SENSOR-1) or (p_1-limits) < 0 or (p_1+limits) > (SIZE_SENSOR-1):
            return intensity
        else:
            intensity = 0
            if rng == 3:
                for i in range(-limits, limits+1):
                    for j in range(-limits, limits+1):
                        index_i = pxl[0] + i
                        index_j = pxl[1] + j
                        if i == 0 and j == 0:
                            intensity = intensity + image[index_i, index_j]*W_PPAL
                        elif i == 0 or j == 0:
                            intensity = intensity + image[index_i, index_j]*W_SIDE
                        else:
                            intensity = intensity + image[index_i, index_j]*W_DIAG
            else:
                for i in range(-limits, limits+1):
                    for j in range(-limits, limits+1):
                        index_i = p_0 + i
                        index_j = p_1 + j
                        intensity = intensity + image[index_i, index_j]
                intensity = intensity/(rng*rng)
            return intensity

    def image_gradient(self, image, ds, pxl_pos, data=True):
        if self.first:
            if data:
                self.first = False

                pxl_pos_0 = np.float32([[pxl_pos[0], pxl_pos[1]]])
                pos_0 = self.p_NIT.project(pxl_pos_0)
                intensity_0 = self.get_value_pixel(image, pxl_pos_0[0])

                self.matrix_intensity[0].append_data(intensity_0)
                self.matrix_pxl_point[0].append_data(pxl_pos_0)
                self.matrix_point[0].append_data(pos_0)
        else:
            self.frame_1 = image

            if data:
                pxl_pos_0 = np.float32([[pxl_pos[0], pxl_pos[1]]])
                pos_0 = self.p_NIT.project(pxl_pos_0)
                intensity_0 = self.get_value_pixel(image, pxl_pos_0[0])

            self.matrix_intensity[0].append_data(intensity_0)
            self.matrix_pxl_point[0].append_data(pxl_pos_0)
            self.matrix_point[0].append_data(pos_0)

            if not self.matrix_intensity[0].complete:
                last_index = self.matrix_point[0].index - 2
                if last_index < NUM_POINTS-1:
                    for x, y in zip(range(last_index, -1, -1), range(last_index+1)):
                        self.get_next_data(x, y, ds)
                else:
                    ly_1 = self.matrix_intensity[NUM_POINTS-2].index - 1
                    for x, y in zip(range(NUM_POINTS-2, -1, -1), range(ly_1, last_index+1)):
                        self.get_next_data(x, y, ds)
            else:
                lx = len(self.matrix_intensity) - 2
                ly_1 = len(self.matrix_intensity[0].data) - 2
                ly_2 = len(self.matrix_intensity[-2].data) - 2
                for x, y in zip(range(lx + 1), range(ly_1, ly_2 - 1, -1)):
                    self.get_next_data(x, y, ds)

    def get_next_data(self, x, y, ds):
        intensity_0 = self.matrix_intensity[x].data[y]
        pos_0 = np.float32(self.matrix_point[x].data[y])
        pxl_pos_0 = np.float32(self.matrix_pxl_point[x].data[y])
        if not np.isnan(intensity_0):
            pos_1, pxl_pos_1, intensity_1 = self.get_next_value(ds, intensity_0, pxl_pos_0, pos_0)
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
            self.ds = vel * dt
        self.time = time
        return self.ds

    def get_gradient(self, rate):
        intensity_0 = self.matrix_intensity[0].data[0]
        intensity_1 = self.matrix_intensity[rate].data[0]
        if intensity_0 == -1 or intensity_1 == -1:
            gradient = np.nan
        else:
            gradient = intensity_1 - intensity_0

    def get_next_value(self, ds, intensity_0, pxl_pos_0, pos=np.float32([[0, 0]])):
        pos_0 = np.float32([[pos[0][0], pos[0][1], 0]])
        pos_1 = pos_0 + ds
        if pos_1 is not None and self.dt < 0.2 * 1000000000:
            pxl_pos_1 = self.p_NIT.transform(pos_1)
            intensity_1 = self.get_value_pixel(self.frame_1, pxl_pos_1[0])
            return pos_1, pxl_pos_1, intensity_1
        else:
            return None, None, None


if __name__ == '__main__':
    from data.analysis import *

    filename = '/home/jorge/Downloads/20_2000.h5'
    data = read_hdf5(filename)
    velocity = calculate_velocity(data['robot'].time, data['robot'].position)
    data['robot'] = append_data(data['robot'], velocity)
    print data

    coolrate = CoolRate()

    files_NIT = "../../../mashes_calibration/data/coolrate/tachyon/image/*.png"
    f_NIT = glob.glob(os.path.realpath(os.path.join(os.getcwd(), files_NIT)))
    coolrate.load_data(
        "../../../mashes_calibration/data/coolrate/velocity/velocity.csv", f_NIT)
