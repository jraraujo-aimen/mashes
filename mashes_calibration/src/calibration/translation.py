class Translation(object):
    def __init__(self, pto_ant, pto_post):
        (self.x1, self.y1) = pto_ant
        (self.x2, self.y2) = pto_post
        self.shift()

    def shift(self,):
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1
