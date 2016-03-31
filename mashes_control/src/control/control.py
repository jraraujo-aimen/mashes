
import yaml
import numpy as np


class Control():
    def __init__(self):
        self.pid = PID()

    def load_conf(self, filename):
        with open(filename, 'r') as f:
            data = yaml.load(f)
        Kp = data['parameters']['Kp']
        Ki = data['parameters']['Ki']
        Kd = data['parameters']['Kd']
        pwr_min = data['power']['min']
        pwr_max = data['power']['max']
        self.pid.setParameters(Kp, Ki, Kd)
        self.pid.setLimits(pwr_min, pwr_max)
        return data

    def save_conf(self, filename):
        Kp, Ki, Kd = self.pid.Kp, self.pid.Ki, self.pid.Kd
        pwr_min, pwr_max = self.pid.pwr_min, self.pid.pwr_max
        data = dict(parameters=dict(Kp=Kp, Ki=Ki, Kd=Kd),
                    power=dict(min=pwr_min, max=pwr_max))
        with open(filename, 'w') as f:
            f.write(yaml.dump(data))
        return data

    def auto_output(self, power_value, set_point):
        k = 1.0033
        power_out = k*(set_point - power_value)
        if power_out > 1500.0:
            power_out = 1500.0
        if power_out < 0.0:
            power_out = 0.0
        return power_out

    def power_detection(self, cv_image):
        mean = np.mean(cv_image[10:20, 10:20, :])
        power = 5.8823 * mean
        return power


class PID():
    def __init__(self):
        self.setParameters(1.0, 1.0, 0.0)
        self.setLimits(0, 1500)
        self.Integrator_max = 10
        self.Integrator_min = -10
        self.set_point = 0.0
        self.error = 0.0

    def setPoint(self, set_point):
        self.set_point = set_point

    def setParameters(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def setLimits(self, pwr_min, pwr_max):
        self.pwr_min = pwr_min
        self.pwr_max = pwr_max

    def update(self, current_value):
        self.error = self.set_point - current_value
        print 'error', self.error
        P = self.Kp * self.error
        print 'Proportional:', P
        I = self.Ki
        print 'Integral:', I
        # self.Integrator = self.Integrator + self.error
        # if self.Integrator > self.Integrator_max:
        #     self.Integrator = self.Integrator_max
        # elif self.Integrator < self.Integrator_min:
        #     self.Integrator = self.Integrator_min
        D = self.Kd  # D_value = self.Kd * (self.error - self.Derivator)
        print 'Derivative', D
        # self.Derivator = self.error
        PID = P + I + D
        print 'PID', PID
        if PID > self.pwr_max:
            PID = self.pwr_max
        if PID < self.pwr_min:
            PID = self.pwr_min
        return PID


if __name__ == '__main__':
    filename = '../../config/control.yaml'
    control = Control()
    control.save_conf(filename)
    print control.load_conf(filename)
