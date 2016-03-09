import yaml
import os
import rospy
import rospkg

class Laser():
    def __init__(self):
    config_file = rospy.get_param('~config', 'labjack.yml')
    path = rospkg.RosPack().get_path('mashes_labjack')
    config_filename = os.path.join(path, 'config', config_file)
    with open(config_filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    self.potencia_max = float(cfg['configuration']['potencia_maxima'])
    self.potencia_min = float(cfg['configuration']['potencia_minima'])
    self.factor = 5.0 / (self.potencia_max - self.potencia_min)
