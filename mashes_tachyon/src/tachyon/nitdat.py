import os
import time
import datetime

import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s')


def thermal_colormap(levels=1024):
    colors = np.array([[0.00, 0.00, 0.00],
                       [0.19, 0.00, 0.55],
                       [0.55, 0.00, 0.62],
                       [0.78, 0.05, 0.55],
                       [0.90, 0.27, 0.10],
                       [0.96, 0.47, 0.00],
                       [1.00, 0.70, 0.00],
                       [1.00, 0.90, 0.20],
                       [1.00, 1.00, 1.00]])
    steps = levels / (len(colors)-1)
    lut = []
    for c in range(3):
        col = []
        for k in range(1, len(colors)):
            col.append(np.linspace(colors[k-1][c], colors[k][c], steps))
        col = np.concatenate(col)
        lut.append(col)
    lut = np.transpose(np.vstack(lut))
    lut_iron = np.uint8(lut * 255)
    return lut_iron

LUT_IRON = thermal_colormap()


class NitDat():
    def __init__(self):
        self.filename = None
        self.N = None
        #Keyhole and ROI
        self.read_config('config.yml')

        self.status = False
        self.background = None
        self.bground = None

    def read_file(self, filename):
        header_dtype = np.dtype([('SW_vers', '<i4'), ('Data_format', '<i2'),
                                 ('Image_rows', '<i2'), ('Image_cols', '<i2'),
                                 ('System_date', '>f8'),
                                 ('Serial_num', '>i4'), ('FW_vers', '>i4'),
                                 ('Conf_regs', ('>i2', 8)),
                                 ('Log_description', 'a2018')])
        with open(filename, 'rb') as f:
            header = np.fromfile(f, dtype=header_dtype, count=1)
            r, c = header['Image_rows'], header['Image_cols']
            data_dtype = np.dtype([('Frame_info', ('>i4', 3)),
                                   ('Frame_data', ('>i2', (r, c)))])
            data = np.fromfile(f, dtype=data_dtype)
        return header, data

    def parse_header(self, header):
        # Data file: test.dat
        # Date of acquisition: 09/01/2015 @ 15:01
        # MATRIX/LUXELL system S/N: 1427967576
        # MATRIX/LUXELL Software rev. 33554688
        # Acquisition software v. 0
        # Config regs: Reg0=0800;Reg1=0000;Reg2=0000;Reg3=0000;
        #              Reg4=0000;Reg5=0000;Reg6=0000;Reg7=0000
        # Integration time used (us): 655,36
        # Reset time used (us): 20,48
        # Image dimensions: 32R x 32C
        # Image format: I16 (2 byte/px)
        # Number of frames in file: 10
        system_date = header['System_date']
        time_offset = time.mktime(datetime.datetime(1904, 1, 1, 0, 0).timetuple())
        print time.strftime('%d/%m/%Y @ %H:%M',
                            time.localtime(system_date + time_offset))

    def compose_header(self, rows, cols):
        #header = np.zeros(2060, dtype=np.uint8)
        header = np.array((0, 2, rows, cols, 0, 2048, 0),
                          dtype=np.dtype([('s', 'a4'),
                                          ('fmt', '<i2'),
                                          ('row', '<i2'),
                                          ('col', '<i2'),
                                          ('v', 'a16'),
                                          ('r0', '>i2'),
                                          ('t', 'a2032')]))
        return header

    def write_file(self, frames, filename):
        header = np.zeros(2060, dtype=np.uint8)
        hframe = np.zeros(12, dtype=np.uint8)
        with open(filename, 'wb') as f:
            header.tofile(f)
            for frame in frames:
                hframe.tofile(f)
                frame.tofile(f)

    def read_config(self, filename='config.yml'):
        with open(filename, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        self.row = cfg['keyhole']['row']
        self.col = cfg['keyhole']['col']
        self.pnt = np.array(cfg['roi']['pnt'])
        self.size = np.array(cfg['roi']['size'])
        logging.debug('Read config: %s' % cfg)

    def write_config(self, filename='config.yml'):
        cfg = {'keyhole': {'row': self.row, 'col': self.col},
               'roi': {'pnt': self.pnt.tolist(),
                       'size': self.size.tolist()}}
        with open(filename, 'w') as ymlfile:
            dump = yaml.dump(cfg, default_flow_style=False)
            ymlfile.write(dump)
        logging.debug('Save config: %s' % cfg)

    def set_filenames(self, filename):
        self.filename = filename
        path, fname = os.path.split(filename)
        name, ext = os.path.splitext(fname)
        self.im1_name = os.path.join(path, name + '.jpg')
        self.im2_name = os.path.join(path, name + 'r.jpg')
        self.dat_name = os.path.join(path, name + '.dat')
        self.lbl_name = os.path.join(path, name + '.npy')
        logging.info('Filename: %sf' % filename)

    def read_frames(self, filename):
        """Reads the frames from a .dat file."""
        frames = None
        if os.path.exists(filename):
            self.set_filenames(filename)
            header, data = self.read_file(filename)
            frames = data['Frame_data']
            frames = (np.int16(frames[1:]) + np.int16(frames[:-1])) >> 1
            h, w = frames.shape[1:]
            self.read_pictures(self.im1_name, self.im2_name)
            self.status = False
            self.N = len(frames)
            logging.info('Total frames: %i' % self.N)
        return frames

    def read_pictures(self, filename1, filename2):
        self.img1 = None
        if os.path.exists(filename1):
            self.img1 = cv2.cvtColor(cv2.imread(filename1), cv2.COLOR_BGR2RGB)
        self.img2 = None
        if os.path.exists(filename2):
            self.img2 = cv2.cvtColor(cv2.imread(filename2), cv2.COLOR_BGR2RGB)
        return self.img1, self.img2

    def read_labels(self, nframes):
        if os.path.exists(self.lbl_name):
            data_labels = np.load(self.lbl_name)
            logging.debug('Labels loaded from file: %s' % self.lbl_name)
        else:
            data_labels = np.empty(nframes)
            data_labels.fill(-1)
        return data_labels

    def write_labels(self, data_labels):
        np.save(self.lbl_name, data_labels)
        logging.debug('Labels saved to file: %s' % self.lbl_name)

    def laser_status(self, keyvals, thr=200):
        if self.status:
            if keyvals[0] - keyvals[-1] > thr:
                self.status = False
        else:
            if keyvals[-1] - keyvals[0] > thr:
                self.status = True
        return self.status

    def update_background(self, frame):
        """dst = (1 - 0.05) * dst + 0.05 * src"""
        if self.bground is None:
            self.bground = np.zeros(frame.shape, dtype=np.float32)
        cv2.accumulateWeighted(np.float32(frame), self.bground, 0.02)
        self.background = np.int16(self.bground)

    def process_frame(self, frame):
        if self.background is None:
            self.background = np.zeros(frame.shape, dtype=np.int16)
        mean = int(cv2.mean(self.background)[0])
        frame = cv2.subtract(frame, self.background - mean)
        frame[frame < 0] = 0
        #frame[frame > 1024] = 1024
        frame = cv2.subtract(np.uint16(frame), mean)
        #frame = cv2.subtract(np.uint16(frame), np.uint16(self.background))
        return np.int16(frame)

    def process_frames(self, frames, offset=1000):
        start, end = self.find_seam(frames)
        self.background = np.int16(np.mean(frames[start-800:start-300], axis=0))
        frames = [self.process_frame(frame) for frame in frames[start:end]]
        return np.int16(frames)

    def find_keyhole(self, frames):
        m, h, w = frames.shape
        idx = np.argmax(frames)
        #idf = idx / h / w
        row = (idx % (h * w)) / h
        col = (idx % (h * w)) % h
        return row, col

    def find_seam(self, frames, offset=0, row=18, col=12, thr=69):
        """Finds the start and the end of the seam using a threshold value."""
        start, end = 0, len(frames)
        if row is None or col is None:
            row, col = self.find_keyhole(frames)
        vals = self.get_frames_pnt(frames, row, col)
        for k in range(3, end-3):
            if vals[k+1] - vals[k-1] > thr:
                start = k
                break
        for k in range(end-3, 3, -1):
            if vals[k-1] - vals[k+1] > thr:
                end = k+1
                break
        start, end = np.clip(np.array([start-offset, end+offset]),
                             0, len(frames))
        logging.debug('Seam start/end (len): %i/%i (%i)'
                      % (start, end, end-start))
        return start, end

    def scale_frame(self, frame):
        frame = cv2.resize(frame, (64, 64))
        return frame

    def get_roi(self, frame):
        (x, y), (w, h) = self.pnt, self.size
        return frame[y:y+h, x:x+w]

    def filter_data(self, filename, out_name=None, offset=1000):
        """Filter irrelevant data stored in the .dat file reducing its size."""
        header, data = self.read_file(filename)
        frames = data['Frame_data'][1:] + data['Frame_data'][:-1]
        start, end = self.find_seam(frames, offset)
        data = data[start:end]
        if out_name is None:
            filename = os.path.splitext(filename)[0]
            filename = filename + 'f.dat'
        with open(filename, 'wb') as f:
            header.tofile(f)
            data.tofile(f)
        logging.info('Saved filename: %s' % filename)

    def get_frames_pnt(self, frames, row=None, col=None):
        if row is None:
            row = self.row
        if col is None:
            col = self.col
        return frames[:, row, col]

    def get_frames_row(self, frames, row=None):
        if row is None:
            row = self.row
        return frames[:, row, :].transpose()

    def get_frames_col(self, frames, col=None):
        if col is None:
            col = self.col
        return frames[:, :, col].transpose()

    def get_frames_roi(self, frames):
        (x, y), (w, h) = self.pnt, self.size
        return frames[:, y:y+h, x:x+w]

    def get_frames_data(self, frames):
        frames_roi = self.get_frames_roi(frames)
        return frames_roi.reshape((len(frames_roi), -1))

    def plot_keyhole(self, frames, row=None, col=None):
        keyvals = self.get_frames_pnt(frames, row, col)
        plt.figure()
        plt.subplot(311)
        plt.plot(np.arange(len(keyvals)), keyvals)
        plt.plot(np.arange(len(keyvals)-4), keyvals[4:] - keyvals[:-4])
        plt.xlim(0, len(keyvals))
        plt.subplot(312)
        img = self.get_frames_col(frames)
        plt.imshow(img, cmap='CMRmap', extent=[0, len(img), 0, len(img)/10])
        plt.subplot(313)
        img = self.get_frames_row(frames)
        plt.imshow(img, cmap='CMRmap', extent=[0, len(img), 0, len(img)/10])
        plt.show()

    def plot_seam(self, frames):
        plt.figure()
        ax1 = plt.subplot2grid((4, 1), (2, 0))
        img = self.get_frames_col(frames)
        ax1.imshow(img, cmap='gnuplot2', extent=[0, len(img), 0, len(img)/12])
        ax2 = plt.subplot2grid((4, 1), (3, 0))
        img = self.get_frames_row(frames)
        ax2.imshow(img, cmap='gnuplot2', extent=[0, len(img), 0, len(img)/12])
        ax2.set_ylabel('Row')
        ax2.get_yaxis().set_ticks([])
        ax3 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
        ax3.imshow(self.img1, extent=[0, len(img), 0, len(img)/5])
        ax3.set_ylabel('Row')
        ax3.get_yaxis().set_ticks([])
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', required=False, type=str)
    args = parser.parse_args()

    if args.data is not None:
        filename = args.data
        NitDat().filter_data(filename)
    else:
        filename = '../../data/69.dat'

    dat = NitDat()
    frames = dat.read_frames(filename)
    dat.plot_keyhole(frames)
    seam = dat.process_frames(frames)

    # for k in range(3, len(frames)):
    #     status = dat.laser_status(frames[k-3:k, dat.row, dat.col])
    # print status

    frame = seam[len(seam)/2]

    plt.figure()
    plt.imshow(LUT_IRON[frame], interpolation='none')
    plt.show()

    img = cv2.cvtColor(LUT_IRON[frame], cv2.COLOR_RGB2BGR)
    cv2.imwrite('colored.png', img)
