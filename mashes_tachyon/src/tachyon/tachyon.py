import os
import cv2
import yaml
import time
import struct
import ctypes
import platform
import subprocess
import numpy as np
from numpy.ctypeslib import ndpointer


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


path = os.path.dirname(os.path.abspath(__file__))
if platform.system() == 'Windows':
    NITdll = ctypes.CDLL(os.path.join(path, 'libtachyon_acq.dll'))
else:
    NITdll = ctypes.cdll.LoadLibrary(os.path.join(path, 'libtachyon_acq.so'))


# Camera management
_open_camera = NITdll.open_camera
_open_camera.restype = ctypes.c_int

_set_active_camera = NITdll.set_active_camera
_set_active_camera.argtypes = [ctypes.c_int]
_set_active_camera.restype = ctypes.c_int

_get_camera_info = NITdll.get_camera_info
_get_camera_info.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_get_camera_info.restype = ctypes.c_int

_close_camera = NITdll.close_camera
_close_camera.restype = ctypes.c_int

_reset_camera = NITdll.reset_camera
_reset_camera.restype = ctypes.c_int

_usb_error = NITdll.usb_error
_usb_error.restype = ctypes.c_char_p

# Camera operation
_start = NITdll.start
_start.restype = ctypes.c_int

_stop = NITdll.stop
_stop.restype = ctypes.c_int

_read_frame = NITdll.read_frame
_read_frame.argtypes = [ndpointer(ctypes.c_uint8), ndpointer(ctypes.c_int16)]
_read_frame.restype = ctypes.c_int

_calibrate = NITdll.calibrate
_calibrate.argtypes = [ctypes.c_int, ctypes.c_int]
_calibrate.restype = ctypes.c_int

_stop_calibration = NITdll.stop_calibration
_stop_calibration.restype = ctypes.c_int

_close_shutter = NITdll.close_shutter
_close_shutter.restype = ctypes.c_int

_open_shutter = NITdll.open_shutter
_open_shutter.restype = ctypes.c_int

# Camera configuration
_set_integration_time = NITdll.set_integration_time
_set_integration_time.argtypes = [ctypes.c_float]
_set_integration_time.restype = ctypes.c_int

_set_wait_time = NITdll.set_wait_time
_set_wait_time.argtypes = [ctypes.c_float]
_set_wait_time.restype = ctypes.c_int

_set_bias = NITdll.set_bias
_set_bias.argtypes = [ctypes.c_float]
_set_bias.restype = ctypes.c_int

_set_vth = NITdll.set_Vth
_set_vth.argtypes = [ctypes.c_int]
_set_vth.restype = ctypes.c_int

_set_timeout = NITdll.set_timeout
_set_timeout.argtypes = [ctypes.c_int]
_set_timeout.restype = ctypes.c_int


class Tachyon():
    def __init__(self, config='tachyon.yml'):
        self.connected = False
        self.config = config

        self.size = None
        self.model = None

        self.frame = None
        self.bground = None
        self.background = None
        self.process_background = False
        
        self.open()

    def open(self):
        """Opens a connection with a TACHYON camera."""
        if not self.connected:
            if _open_camera() > 0:  # Returns the number of cameras
                self.connected = True
                self.model = self.configure_bitstream()
                if self.model is not None:
                    self.configure(self.config)
                else:
                    self.close()
            else:
                print 'ERROR: TACHYON camera not connected.'
        return self.connected

    def close(self):
        """Closes a connection with a TACHYON camera."""
        if self.connected:
            if _close_camera() > 0:
                time.sleep(0.1)
                self.connected = False
        return not self.connected

    def get_camera_info(self):
        """Returns camera info: description, serial, and manufacturer."""
        description = ctypes.create_string_buffer(256)
        serial_number = ctypes.create_string_buffer(256)
        manufacturer = ctypes.create_string_buffer(256)
        _get_camera_info(description, serial_number, manufacturer)
        return description.value, serial_number.value, manufacturer.value

    def configure_bitstream(self):
        camera_info = self.get_camera_info()
        if camera_info[0] == 'TACHYON_1024_microCORE':
            self.size = 32
            self.model = 'microcore'
        elif camera_info[0] == 'TACHYON 1024 NEW_INFRARED_TECHN':
            subprocess.call(['java', '-cp', 'FWLoader.jar',
                             'FWLoader', '-c', '-uf', 'nit_tachyon_32_HS.bit'])
            self.size = 32
            self.model = 'tachyon1024'
        elif camera_info[0] == 'TACHYON 6400 NEW_INFRARED_TECHN':
            self.size = 80
            self.model = 'tachyon6400'
        print camera_info, self.model, self.size
        return self.model

    def set_configuration(self, int_time, wait_time, bias, vth_value, timeout):
        if _set_integration_time(int_time) == 0:
            self.int_time = int_time
        if _set_wait_time(wait_time) == 0:
            self.wait_time = wait_time
        if _set_bias(bias) == 0:
            self.bias = bias
        if _set_vth(vth_value) == 0:
            self.vth_value = vth_value
        if _set_timeout(timeout) == 0:
            self.timeout = timeout

    def set_integration_time(self, int_time):
        self.set_configuration(int(int_time), int(1000-int_time),
                               self.bias, self.vth_value, self.timeout)

    def configure(self, filename):
        with open(filename, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        print cfg
        self.set_configuration(cfg['int_time'], cfg['wait_time'],
                               cfg['bias'], cfg['vth_value'], cfg['timeout'])

    def calibrate(self, target=100, auto_off=1):
        """Performs an offset calibration for dark current correction."""
        for k in range(3000):
            image, header = self.read_frame()
            if k == 0:
                _close_shutter()
            elif k == 550:
                _calibrate(target, auto_off)
            elif k == 2000:
                _stop_calibration()
            elif k > 2050 and k < 2500:
                self.update_background(image)
            elif k == 2500:
                _open_shutter()

    def start_calibration(self, target=100, auto_off=1):
        _close_shutter()
        time.sleep(0.5)
        _calibrate(target, auto_off)
        time.sleep(0.1)

    def stop_calibration(self):
        _stop_calibration()
        time.sleep(0.1)
        self.process_background = True
        time.sleep(1)
        self.process_background = False
        time.sleep(0.1)
        _open_shutter()
        time.sleep(0.5)

    def connect(self):
        if self.connected:
            self.flush_buffer()
            init = _start()
            if not init:
                _open_shutter()
                time.sleep(0.5)
                return True
            else:
                return False
        return False

    def disconnect(self):
        if self.connected:
            end = _stop()
            if end < 0:
                return False
            else:
                _close_shutter()
                time.sleep(0.5)
                return True
        return False

    def flush_buffer(self):
        self.disconnect()
        frame = ()
        while frame is not None:
            frame = self.read_frame()
        time.sleep(0.5)

    def read_frame(self):
        header = np.zeros(64, dtype=np.uint8)
        image = np.zeros(self.size * self.size, dtype=np.int16)
        i = _read_frame(header, image)
        if i < 0:
            if i == -116:
                print 'ERROR: (Timeout) Waiting for images.'
            return None
        else:
            return np.uint16(image.reshape(self.size, self.size)), header

    def parse_header(self, header):
        if self.model == 'microcore':
            # uCORE
            header_type = np.dtype([('Header_ID', '<u4'),
                                    ('Frame_counter', 'a4'),
                                    ('Not_used0', '<u4'),
                                    ('Miliseconds_counter', '<u4'),
                                    ('Buffer_status', 'a4'),
                                    ('Last_command', 'a4'),
                                    ('Integration_time', '<u2'),
                                    ('Wait_time', '<u2'),
                                    ('Shutter_status', '<u2'),
                                    ('Bias_status', '<u2'),
                                    ('Reserved', 'a8'),
                                    ('Calibration_status', '<u2'),
                                    ('Temperature', '<u2'),
                                    ('Not_used1', 'a20')])
        else:
            # CORE
            header_type = np.dtype([('Header_ID', '<u4'),
                                    ('Frame_counter', 'a4'),
                                    ('Microseconds_counter_high', '<u4'),
                                    ('Microseconds_counter_low', '<u4'),
                                    ('Buffer_status', 'a4'),
                                    ('Reserved', 'a4'),
                                    ('Status', 'a4'),
                                    ('Bias_status', 'a4'),
                                    ('REGS', 'a32')])
        parsed_header = header.view(dtype=header_type)[0]
        parsed_header['Frame_counter'] = struct.unpack('I', struct.pack(
            'BBBB', header[6], header[7], header[4], header[5]))[0]
        return parsed_header

    def update_background(self, frame):
        """dst = (1 - 0.05) * dst + 0.05 * src"""
        if self.bground is None:
            self.bground = np.zeros(frame.shape, dtype=np.float32)
        cv2.accumulateWeighted(np.float32(frame), self.bground, 0.02)
        self.background = np.uint16(self.bground)

    def process_frame(self, frame):
        if self.background is None:
            self.background = np.zeros(frame.shape, dtype=np.uint16)
        if self.frame is not None:
            mframe = (frame + self.frame) >> 1
        else:
            mframe = frame
        self.frame = frame
        return cv2.subtract(mframe, self.background)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.set_printoptions(formatter={'int': hex})

    tachyon = Tachyon(config='../../config/tachyon.yml')
    #tachyon.set_integration_time(200)
    print tachyon.int_time, tachyon.wait_time
    tachyon.connect()

    tachyon.calibrate(32)
    for k in range(2500):
        image, header = tachyon.read_frame()
        image = tachyon.process_frame(image)
        print tachyon.parse_header(header)#['Temperature'],
    tachyon.disconnect()

    print tachyon.parse_header(header)

    print 'I', np.mean(image), 'B', np.mean(tachyon.background)

    plt.figure()
    plt.subplot(121)
    plt.imshow(LUT_IRON[image], interpolation='none')
    plt.subplot(122)
    plt.imshow(LUT_IRON[tachyon.background], interpolation='none')
    plt.show()

    tachyon.close()
