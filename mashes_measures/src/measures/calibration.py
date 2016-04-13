import numpy as np


class Calibration():
    def __init__(self, scale=0.375):
        self.scale = scale

    def correct(self, value):
        return self.scale * value


if __name__ == '__main__':
    calibration = Calibration()
    print calibration.correct(10.0)
