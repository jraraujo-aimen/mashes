#!/usr/bin/env python
import cv2
import numpy as np

from measures.projection import Homography
import matplotlib.pyplot as plt
from scipy import stats


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
NUM_POINTS_FITTED = 7

COLORS = ('b', 'g',  'y', 'r', 'm', 'c', 'k')

THR_NO_LASER = 40
LASER_THRESHOLD = 400
W_PPAL = 0.9
W_SIDE = 0.015
W_DIAG = 0.01
FRAME_SAMPLE = 46


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


#-----------------------------additional classes------------------------------#

class CoolRate():
    def __init__(self):

        self.points_plotted = 0

        self.start = False
        self.laser_on = False

        self.hom = Homography()
        self.hom.load('../../../mashes_calibration/config/tachyon_NO.yaml')

        self.max_value = []
        self.max_i = []
        self.min_value = []
        self.min_i = []

        self.time = None
        self.dt = None
        self.ds = None

        self.first = True    # ok
        self.counter = 0

        self.frame_0 = np.zeros((SIZE_SENSOR, SIZE_SENSOR, 1), dtype=np.uint8)
        self.frame_1 = np.zeros((SIZE_SENSOR, SIZE_SENSOR, 1), dtype=np.uint8)
        self.image = np.zeros((SIZE_SENSOR, SIZE_SENSOR, 1), dtype=np.uint8)

        self.sizes = list(range(
            INIT_POINTS + NUM_POINTS, INIT_POINTS, -1))
        self.matrix_intensity = [RingBuffer(a) for a in self.sizes]
        self.matrix_point = [RingBuffer(a) for a in self.sizes]
        self.matrix_pxl_point = [RingBuffer(a) for a in self.sizes]
        self.total_t = RingBuffer(NUM_POINTS)
        self.dt_axis = []
        self.t_axis = []
        self.total_intensity = []

        self.coeff_1 = []
        self.coeff_2 = []
        self.coeff_3 = []

    def fit_exp_linear(self, t, y, C=0):
        y = y - C
        y = np.log(y)
        K, A_log = np.polyfit(t, y, 1)
        A = np.exp(A_log)
        return A, K

    def model_func(self, t, A, K, C):
        return A * np.exp(K * t) + C

    def get_maxvalue(self, frame, rng=3):
        image = np.zeros((SIZE_SENSOR, SIZE_SENSOR), dtype=np.uint16)
        pxls = [np.float32([v, u]) for v in range(0, SIZE_SENSOR) for u in range(0, SIZE_SENSOR)]
        for pxl in pxls:
            index = pxl[0], pxl[1]
            intensity = self.get_value_pixel(frame, pxl)
            if intensity == -1:
                image[index] = 0
            else:
                image[index] = intensity

        value = np.amax(image)
        i = np.unravel_index(np.argmax(image), image.shape)
        print value, i
        if value > LASER_THRESHOLD:
            print "laser on"
            return i
        else:
            return None

    def get_value_pixel(self, frame, pxl, rng=3):
        intensity = -1
        limits = (rng - 1)/2
        p_0 = round(pxl[0])
        p_1 = round(pxl[1])
        if (((p_0-limits) < 0) or ((p_0+limits) > (SIZE_SENSOR-1))
                or ((p_1-limits) < 0) or ((p_1+limits) > (SIZE_SENSOR-1))):
            return intensity
        else:
            intensity = 0
            if rng == 3:
                for v in range(-limits, limits+1):
                    for u in range(-limits, limits+1):
                        index_v = pxl[0] + v
                        index_u = pxl[1] + u
                        if u == 0 and v == 0:
                            intensity = intensity + frame[index_v, index_u]*W_PPAL
                        elif u == 0 or v == 0:
                            intensity = intensity + frame[index_v, index_u]*W_SIDE
                        else:
                            intensity = intensity + frame[index_v, index_u]*W_DIAG
            else:
                for v in range(-limits, limits+1):
                    for u in range(-limits, limits+1):
                        index_v = p_0 + v
                        index_u = p_1 + u
                        intensity = intensity + frame[index_v, index_u]
                intensity = intensity/(rng*rng)
            return intensity

    def image_gradient(self, robot, index_robot, frame, pxl_pos, time, data=True):
        velocity = robot['velocity']
        if self.first:
            if data:
                self.first = False
                vel = velocity.iloc[index_robot]
                self.get_ds(time, vel)
                self.frame_0 = frame

                pxl_pos_0 = np.float32([[pxl_pos[0], pxl_pos[1]]])
                pos_0 = self.hom.transform(pxl_pos_0)
                intensity_0 = self.get_value_pixel(self.frame_0, pxl_pos_0[0])

                self.matrix_intensity[0].append_data(intensity_0)
                self.matrix_pxl_point[0].append_data(pxl_pos_0)
                self.matrix_point[0].append_data(pos_0)
        else:
            vel = velocity.iloc[index_robot]
            self.get_ds(time, vel)
            self.frame_1 = self.frame_0
            self.frame_0 = frame

            if data:
                pxl_pos_0 = np.float32([[pxl_pos[0], pxl_pos[1]]])
                print "pxl_pos_0", pxl_pos_0
                pos_0 = self.hom.transform(pxl_pos_0)
                print "pos_0", pos_0
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
                if last_index < NUM_POINTS-1:
                    for x, y in zip(range(last_index, -1, -1), range(last_index+1)):
                        self.get_next_data(x, y)
                else:
                    ly_1 = self.matrix_intensity[NUM_POINTS-2].index - 1
                    for x, y in zip(range(NUM_POINTS-2, -1, -1), range(ly_1, last_index+1)):
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
            pos_1, pxl_pos_1, intensity_1 = self.get_next_value(intensity_0, pxl_pos_0, pos_0)
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
            self.ds = vel * dt * 1000
            #m/s
            #/ 1000000000
        self.time = time

    def get_next_value(self, intensity_0, pxl_pos_0, pos=np.float32([[0, 0]])):
        pos_0 = np.float32([[pos[0][0], pos[0][1], 0]])
        pos_1 = self.get_position(pos_0)
        if pos_1 is not None and self.dt < 0.2 * 1000000000:
        #         #frame_0: position_0
            pxl_pos_1 = self.hom.project(pos_1)
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

    def find_robot(self, robot, times):
        #indice de los datos del robot con el que se corresponde
        #cada medida de la tachyon tras aplicar una frecuencia de muestreo
        times_robot = np.array(robot.time)
        d_times = [abs(x - times_robot) for x in times]
        robot_index = [y.tolist().index(min(y)) for y in d_times]
        return robot_index

    def plot_fitted_data(self, x_t):
        # print self.coeff_1, self.coeff_2, x_t
        fig_1 = plt.figure()
        fig_1.suptitle('Fitted Function:\n y = A e^(K t) + 0')
        ax1 = fig_1.add_subplot(211)
        ax1.plot(x_t, self.coeff_1, linewidth=LINE_WIDTH)
        ax1.set_title('A', fontsize=FONT_SIZE)
        ax1.set_xlabel('ms', fontsize=FONT_SIZE)
        ax1.set_ylabel('value', fontsize=FONT_SIZE)
        ax1.tick_params(labelsize=FONT_SIZE)
        ax1.set_ylim([0, 2200])
        ax2 = fig_1.add_subplot(212)
        ax2.plot(x_t, self.coeff_2, linewidth=LINE_WIDTH)
        ax2.set_title('K', fontsize=FONT_SIZE)
        ax2.set_xlabel('ms', fontsize=FONT_SIZE)
        ax2.set_ylabel('value', fontsize=FONT_SIZE)
        ax2.tick_params(labelsize=FONT_SIZE)
        ax2.set_ylim([-1.2, 0])
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

    def visualize(self, axisNum,ax1_0, intensity, frame):
        img = frame.copy()
        img = LUT_IRON[img]
        self.total_intensity.append(intensity)
        x = np.arange(NUM_POINTS).tolist()
        x_a = np.array(x)
        intensity_f = np.array(intensity[:NUM_POINTS_FITTED])
        x_f = np.array(np.arange(NUM_POINTS_FITTED).tolist())

        A, K = coolrate.fit_exp_linear(x_f, intensity_f, THR_NO_LASER)
        fit_y = coolrate.model_func(x_a, A, K, THR_NO_LASER)
        coolrate.coeff_1.append(A)
        coolrate.coeff_2.append(K)
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

        w, h, c = img.shape
        img_plus = cv2.resize(img, (w*SCALE_VIS, h*SCALE_VIS), interpolation=cv2.INTER_LINEAR)
        pxl = [coolrate.matrix_pxl_point[t].data[0] for t in range(NUM_POINTS)]
        for p in pxl:
            if not np.isnan(p).any():
                print p
                cv2.circle(img_plus, (int(round(p[0][1])*SCALE_VIS), int(round(p[0][0])*SCALE_VIS)), 4, WHITE, -1)
        cv2.imshow("Image: ", img_plus)
        cv2.waitKey(PAUSE_IMAGE)
        coolrate.t_axis.append(float(coolrate.total_t.data[0])/1000000)

        return axisNum

    def define_graphs(self):
        #inicializalizo la figura donde se van a visualizar las graficas
        fig_0 = plt.figure()
        fig_0.suptitle('Data and Fitted Function')
        ax1_0 = fig_0.add_subplot(111)
        ax1_0.set_xlabel('ms', fontsize=FONT_SIZE)
        ax1_0.set_ylabel('counts', fontsize=FONT_SIZE)
        ax1_0.set_xlim([0, FRAME_SAMPLE*(NUM_POINTS-1)])
        ax1_0.set_ylim([0, 1024])
        ax1_0.tick_params(labelsize=FONT_SIZE)
        return ax1_0



if __name__ == '__main__':

    from tachyon.tachyon import LUT_IRON
    from data.analysis import *
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file', type=str, default=None, help='hdf5 filename')
    args = parser.parse_args()

    filename = args.file

    data = read_hdf5(filename)
    if 'robot' in data.keys():
        velocity_data = calculate_velocity(data['robot'].time, data['robot'].position)
        data['robot'] = append_data(data['robot'], velocity_data)
        robot = data['robot']
    if 'tachyon' in data.keys():
        tachyon = data['tachyon']
        tachyon = tachyon[tachyon.frame.notnull()]
        idx = np.arange(0, len(tachyon), FRAME_SAMPLE)
        times = np.array(tachyon.time[idx])
        frames = read_frames(tachyon.frame[idx])

    coolrate = CoolRate()
    robot_index = coolrate.find_robot(robot, times)

    axisNum = 0

    # #inicializalizo la figura donde se van a visualizar las graficas
    ax1_0 = coolrate.define_graphs()

    plt.ion()
    for (frame, time, index_robot) in zip(frames, times, robot_index):

        print index_robot
        y = x = np.arange(0, SIZE_SENSOR, 1)
        Y, X = np.meshgrid(y, x)
        index_pixel = coolrate.get_maxvalue(frame)

        if index_pixel is not None:
            #calcula gradiente
            coolrate.image_gradient(robot, index_robot, frame, index_pixel,
                                    time, True)
        else:
            if coolrate.matrix_intensity[0].stored_data:
                coolrate.image_gradient(robot, index_robot, frame, index_pixel,
                                        time, False)

        if coolrate.matrix_intensity[0].index == coolrate.matrix_intensity[0].length:
            intensity = [coolrate.matrix_intensity[point].data[0] for point in range(NUM_POINTS)]
            axisNum = coolrate.visualize(axisNum, ax1_0, intensity, frame)

    x_t = [coolrate.t_axis[t] - coolrate.t_axis[0] for t in range(len(coolrate.t_axis))]
    coolrate.plot_fitted_data(x_t)

    coolrate.print_stats()

    while True:
        plt.pause(PAUSE_PLOT)
