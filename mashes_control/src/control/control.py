
import cv2
import numpy as np


class Control():
    def __init__(self):
        pass

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
    """
    Discrete PID control
    """

    def __init__(self, P=1.0, I=1.0, D=1.0, Derivator=0, Integrator=0,
                 Integrator_max=10, Integrator_min=-10):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.Derivator = Derivator
        self.Integrator = Integrator
        self.Integrator_max = Integrator_max
        self.Integrator_min = Integrator_min
        self.set_point = 0.0
        self.error = 0.0

    def update(self, current_value):
        """
        Calculate PID output value for given reference input and feedback
        """
        self.error = self.set_point - current_value
        print 'current_value'
        print current_value
        print 'error'
        print self.error
        self.P_value = self.Kp * (self.error)
        print 'parte proporcional'
        print self.P_value
        self.D_value = self.Kd * (self.error - self.Derivator)
        print 'parte derivativa'
        print self.D_value
        self.Derivator = self.error
        self.Integrator = self.Integrator + self.error
        if self.Integrator > self.Integrator_max:
            self.Integrator = self.Integrator_max
        elif self.Integrator < self.Integrator_min:
            self.Integrator = self.Integrator_min

        self.I_value = self.Integrator * self.Ki
        print 'parte integradora'
        print self.I_value

        PID = self.P_value + self.I_value + self.D_value
        print 'valor final'
        print PID
        if PID > 1500:
            PID = 1500
        if PID < 0:
            PID = 0
        return PID

    def setPoint(self, set_point):
        self.set_point = set_point
        self.Integrator = 0
        self.Derivator = 0

    def setIntegrator(self, Integrator):
        self.Integrator = Integrator

    def setDerivator(self, Derivator):
        self.Derivator = Derivator

    def setKp(self, P):
        self.Kp = P

    def setKi(self, I):
        self.Ki = I

    def setKd(self, D):
        self.Kd = D

    def getPoint(self):
        return self.set_point

    def getError(self):
        return self.error

    def getIntegrator(self):
        return self.Integrator

    def getDerivator(self):
        return self.Derivator

if __name__ == '__main__':
    control = Control()
