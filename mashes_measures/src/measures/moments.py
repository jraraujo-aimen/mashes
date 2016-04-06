import numpy as np


class Moments():
    def __init__(self, threshold=127):
        self.threshold = threshold

    def binarize(self, frame):
        img_bin = np.zeros(frame.shape, dtype=np.uint8)
        img_bin[frame > self.threshold] = 255
        return img_bin

    def levels(self, img, min=0, max=255):
        lut = np.arange(256)
        for k in lut:
            if k <= min:
                lut[k] = 0
            elif k > min and k < max:
                lut[k] = (255. / (max - min)) * (k - min)
            else:
                lut[k] = 255
        lut_levels = np.uint8(lut)
        print lut_levels
        img = lut_levels[img]
        # self.lut_conv = (np.asarray(range(value))*(256))/value
        # img=self.lut_conv[img]
        return img

    def bounding_box(self, image):
        size = 0
        width, height = image.shape[1], image.shape[0]
        # Boundary box
        left, top = width, height
        right, bottom = 0, 0
        for y in range(height):
            for x in range(width):
                if image[y, x]:
                    size = size + 1
                    if left > x:
                        left = x
                    if top > y:
                        top = y
                    if right < x:
                        right = x
                    if bottom < y:
                        bottom = y
        if size:
            box = [left, top, right, bottom]
        else:
            box = [0, 0, 0, 0]
        return box

    def moments(self, image):
        width, height = image.shape[1], image.shape[0]
        m00, m01, m10, m11, m02, m20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for y in range(height):
            for x in range(width):
                m00 = m00 + image[y, x]
                m10 = m10 + x * image[y, x]
                m01 = m01 + y * image[y, x]
                m11 = m11 + x * y * image[y, x]
                m20 = m20 + x * x * image[y, x]
                m02 = m02 + y * y * image[y, x]
        return [m00, m01, m10, m11, m02, m20]

    def ellipse(self, image):
        """Calculates the ellipse approximation of the shape in the image."""
        m00, m01, m10, m11, m02, m20 = self.moments(image)
        x_cm, y_cm, angle, length, width = 0, 0, 0, 0, 0
        if m00 > 1000:
            x_cm = m10 / m00
            y_cm = m01 / m00
            # Rounded
            m11 = m11 / 64 * 64
            m20 = m20 / 64 * 64
            m02 = m02 / 64 * 64
            # ---
            u20 = float(m20 / m00 - x_cm * x_cm)
            u02 = float(m02 / m00 - y_cm * y_cm)
            u11 = float(m11 / m00 - x_cm * y_cm)
            lmax = (u20 + u02 + np.sqrt(4 * u11 * u11 + (u20 - u02) * (u20 - u02))) / 2
            lmin = (u20 + u02 - np.sqrt(4 * u11 * u11 + (u20 - u02) * (u20 - u02))) / 2
            # ---
            angle = np.arctan((2 * u11) / (u20 - u02)) / 2
            length = 4 * np.sqrt(lmax)
            width = 4 * np.sqrt(lmin)
        return [x_cm, y_cm, angle, length, width]

    def find_geometry(self, frame):
        img_bin = self.binarize(frame)
        x_cm, y_cm, angle, length, width = self.ellipse(img_bin)
        major_axis, minor_axis, angle_rads = length, width, angle
        return major_axis, minor_axis, angle_rads


if __name__ == '__main__':
    import cv2
    from pylab import *
    from matplotlib.patches import Rectangle, Ellipse

    img = cv2.imread('../../data/frame0015.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = (img.astype("uint16"))*(1024/255)
    #print np.max(img)

    moments = Moments(127)
    img_bin = moments.binarize(img)
    x_cm, y_cm, angle, length, width = moments.ellipse(img_bin)
    print width, length, angle

    img_pix = moments.levels(img, 63, 191)
    x_cm_pixel, y_cm_pixel, angle_pixel, length_pixel, width_pixel = moments.ellipse(img_pix)
    print width_pixel, length_pixel, angle_pixel

    left, top, right, bottom = moments.bounding_box(img)
    print right - left, bottom - top

    figure()
    subplot(221)
    imshow(img_bin, cmap='gray')
    axis('off')
    subplot(222)
    imshow(img_pix, cmap='gray')
    axis('off')
    subplot(223)
    imshow(img_bin, cmap='gray')
    axis('off')
    gca().add_patch(Ellipse([x_cm, y_cm], length, width,
                            angle=rad2deg(angle),
                            facecolor='none',
                            edgecolor='red', lw=1.5))
    subplot(224)
    imshow(img_pix, cmap='gray')
    axis('off')
    gca().add_patch(Ellipse([x_cm_pixel, y_cm_pixel],
                            length_pixel, width_pixel,
                            angle=rad2deg(angle_pixel),
                            facecolor='none', edgecolor='red', lw=1.5))
    show()
