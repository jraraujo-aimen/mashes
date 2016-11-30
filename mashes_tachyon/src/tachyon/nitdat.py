import os
import time
import datetime

import cv2
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

        self.frame = None
        self.bground = None
        self.background = None

    def read_file(self, filename):
        header_dtype = np.dtype([('SW_vers', '<i4'), ('Data_format', '<i2'),
                                 ('Image_rows', '<i2'), ('Image_cols', '<i2'),
                                 ('System_date', '>f8'),
                                 ('Serial_num', '<i4'), ('FW_vers', '>i4'),
                                 ('Conf_regs', ('>i2', 8)),
                                 ('Log_description', 'a2018')])
        with open(filename, 'rb') as f:
            header = np.fromfile(f, dtype=header_dtype, count=1)
            r, c = header['Image_rows'], header['Image_cols']
            data_dtype = np.dtype([('Frame_info', ('>i2', 6)),
                                   ('Frame_data', ('>i2', (r, c)))])
            data = np.fromfile(f, dtype=data_dtype)
        self.parse_header(header)
        return header, data

    def parse_header(self, header):
        time_offset = time.mktime(datetime.datetime(1904, 1, 1, 0, 0).timetuple())
        print 'Data file:'
        print 'Date of acquisition:', time.strftime('%d/%m/%Y @ %H:%M',
                                                    time.localtime(header['System_date'] + time_offset))
        print 'MATRIX/LUXELL system S/N:', header['Serial_num'][0]
        print 'MATRIX/LUXELL Software rev.', header['FW_vers'][0]
        print 'Config regs:', header['Conf_regs'][0]
        print 'Integration time used (us):'
        print 'Reset time used (us):'
        print 'Image dimension:'
        print 'Image format:', header['Data_format'][0]
        print 'Number of frames in file:'

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

    def read_frames(self, filename):
        frames = None
        if os.path.exists(filename):
            self.filename = filename
            header, data = self.read_file(filename)
            print 'Frame_info:', data['Frame_info']
            frames = np.uint16(data['Frame_data'])
            frames = (frames[1:] + frames[:-1]) >> 1
            self.N = len(frames)
            logging.info('File: %s Frames: %i' % (self.filename, self.N))
        return frames

    def write_frames(self, filename, frames):
        rows, cols = frames[0].shape
        header = self.compose_header(rows, cols)
        hframe = np.zeros(12, dtype=np.uint8)
        with open(filename, 'wb') as f:
            header.tofile(f)
            for n, frame in enumerate(frames):
                hframe = np.array([300, n, n * 100], dtype=np.dtype('>i4'))
                frame = frame.astype(np.dtype('>i2'))
                hframe.tofile(f)
                frame.tofile(f)

    def update_background(self, frame):
        """dst = (1 - 0.05) * dst + 0.05 * src"""
        if self.bground is None:
            self.bground = np.zeros(frame.shape, dtype=np.float32)
        cv2.accumulateWeighted(np.float32(frame), self.bground, 0.02)
        self.background = np.int16(self.bground)

    def process_frame(self, frame):
        if self.background is None:
            self.background = np.zeros(frame.shape, dtype=np.int16)
        frame = cv2.subtract(np.uint16(frame), np.uint16(self.background))
        return np.int16(frame)

    def nuc_correction(self, frame):
        gain = np.ones(frame.shape)
        bias = np.zeros(frame.shape)
        frame = gain * frame + bias
        return np.int16(frame)

    def scale_frame(self, frame):
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (2*w, 2*h))
        return frame


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', '-d', type=str, default='../../data/test.dat')
    args = parser.parse_args()

    filename = args.data

    dat = NitDat()
    frames = dat.read_frames(filename)
    #dat.write_frames('../../data/test.dat', frames)

    for frame in frames[50:250]:
        dat.update_background(frame)
    #frames = np.int16([dat.process_frame(frame) for frame in frames])

    frame = frames[len(frames)/2]
    pframe = dat.process_frame(frame)
    pframe = dat.nuc_correction(pframe)
    #pframe = dat.scale_frame(pframe)

    plt.figure()
    plt.subplot(221)
    plt.imshow(LUT_IRON[dat.background], interpolation='none')
    plt.subplot(222)
    plt.imshow(LUT_IRON[frame], interpolation='none')
    plt.subplot(223)
    plt.imshow(LUT_IRON[pframe], interpolation='none')
    plt.show()

    #cv2.imwrite('colored.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    imean = np.mean(frames[0:100])
    istd = np.std(frames[0:100])
    print imean, istd
