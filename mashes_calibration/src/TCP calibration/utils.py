import cv2
import numpy as np
import math

class Utils(object):
    def __init__(self):
        self.A_1 = 0
        self.A_2 = 0
        self.B_1 = 0
        self.B_2 = 0
        self.C_1 = 0
        self.C_2 = 0
        self.point_1 = 0
        self.point_2 = 0
        self.point_3 = 0

        self.STEPS = 1

    def mouse_handler(self, event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(data['points']) < 4:
                cv2.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16)
                cv2.imshow("Image", data['im'])
                len_points = len(data['points'])
                if len_points == 0:
                    self.point_1 = (x, y)
                elif len_points == 1:
                    self.point_2 = (x, y)
                elif len_points == 2:
                    self.point_3 = (x, y)
                elif len_points == 3:
                    self.origin_vectors(self.point_1, self.point_2, self.point_3)
                data['points'].append([x, y])
            else:
                pass


    def mouse_handler_two(self, event, x, y, flags, data_two):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(data_two['points']) < 2:
                cv2.circle(data_two['im'], (x, y), 3, (0, 0, 255), 5, 16)
                cv2.imshow("Image", data_two['im'])
                len_points = len(data_two['points'])
                if len_points == 0:
                    self.point_1 = (x, y)
                elif len_points == 1:
                    self.point_2 = (x, y)
                data_two['points'].append([x, y])
            else:
                pass

    def mouse_handler_one(self, event, x, y, flags, data_one):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(data_one['points']) < 1:
                cv2.circle(data_one['im'], (x, y), 3, (0, 0, 255), 5, 16)
                cv2.imshow("Image", data_one['im'])
                len_points = len(data_one['points'])
                if len_points == 0:
                    self.point_1 = (x, y)
                data_one['points'].append([x, y])
            else:
                pass

    def origin_vectors(self, point_1, point_2, point_3):
        self.A_1 = point_2[1] - point_1[1]
        self.B_1 = point_1[0] - point_2[0]
        self.C_1 = point_1[1]*(point_2[0]-point_1[0]) - point_1[0]*(point_2[1] - point_1[1])

        self.A_2 = point_3[1] - point_1[1]
        self.B_2 = point_1[0] - point_3[0]
        self.C_2 = point_1[1]*(point_3[0]-point_1[0]) - point_1[0]*(point_3[1] - point_1[1])


    def Point2CoordLines(self, point):

        den_1 = point[0]*self.A_1 + point[1]*self.B_1 + self.C_1
        if den_1 < 0:
            den_1 = den_1*(-1)
        num_1 = math.sqrt(self.A_1**2 + self.B_1**2)
        if den_1 == 0:
            y = 0
        else:
            y = den_1/num_1

        den_2 = point[0]*self.A_2 + point[1]*self.B_2 + self.C_2
        if den_2 < 0:
            den_2 = den_2*(-1)
        num_2 = math.sqrt(self.A_2**2 + self.B_2**2)
        if den_2 == 0:
            x = 0
        else:
            x = den_2/num_2

        return x, y

    def get_four_points(self, im):

        # Set up data to send to mouse handler
        data = {}
        data['im'] = im.copy()
        data['points'] = []
        coord_points = []

        #Set the callback function for any mouse event
        cv2.imshow("Image", im)
        cv2.setMouseCallback("Image", self.mouse_handler, data)
        cv2.waitKey(0)

        # Convert array to np.array
        points = np.vstack(data['points']).astype(float)
        # return points
        for point in points:
            x, y = self.Point2CoordLines(point)
            if len(coord_points) < 4:
                coord_points.append([x, y])
        coord_points = np.vstack(coord_points).astype(float)
        return np.float32(points)


    def get_two_points(self, im):

        # Set up data to send to mouse handler
        data_two = {}
        data_two['im'] = im.copy()
        data_two['points'] = []
        #Set the callback function for any mouse event
        cv2.imshow("Image", im)
        cv2.setMouseCallback("Image", self.mouse_handler_two, data_two)
        cv2.waitKey(0)

        # Convert array to np.array
        points = np.vstack(data_two['points']).astype(float)
        return points

    def get_point(self, im):

        # Set up data to send to mouse handler
        data_one = {}
        data_one['im'] = im.copy()
        data_one['points'] = []
        #Set the callback function for any mouse event
        cv2.imshow("Image", im)
        cv2.setMouseCallback("Image", self.mouse_handler_one, data_one)
        cv2.waitKey(0)

        # Convert array to np.array
        points = np.vstack(data_one['points']).astype(float)
        return points

    def points_average(self, image, points=4):
        pts = []
        print '''
            Click on the four/two/one corner/s of the pattern -- top left first and
            bottom rigth last -- and then hit ENTER
            '''
        if points == 4:
            pts_ptr = self.get_four_points(image)
        elif points == 2:
            pts_ptr = self.get_two_points(image)
        elif points == 1:
            pts_ptr = self.get_point(image)

        pts.append(pts_ptr)
        for i in range(self.STEPS - 1):
            print '''
                Repit selection one more time -- and then hit ENTER
                '''
            if points == 4:
                pts_ptr = self.get_four_points(image)
            elif points == 2:
                pts_ptr = self.get_two_points(image)
            elif points == 1:
                pts_ptr = self.get_point(image)

            pts.append(pts_ptr)

        pts_mean = np.mean(pts, axis=0)
        return pts_mean
