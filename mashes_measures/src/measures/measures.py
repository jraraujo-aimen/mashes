#!/usr/bin/env python

import time 
from pylab import *
from PIL import Image
from matplotlib.patches import Rectangle, Ellipse

def binarize(image, threshold):
    """Binarizes the image through the configured thresholds."""
    width, height = image.shape[1], image.shape[0]
    threshold = uint8(threshold)
    img = zeros((height, width), dtype=uint8)
    for y in range(height):
        for x in range(width):
            if image[y,x] > threshold:
                img[y,x] = 255
    return img

def erosion(image):
    """Erodes the image with a 3x3 kernel."""
    width, height = image.shape[1], image.shape[0]
    img = zeros((height, width), dtype=uint8)
    for y in range(height - 2):
        for x in range(width - 2):
            if (image[y,x] and image[y,x+1] and image[y,x+2] and
                image[y+1,x] and image[y+1,x+1] and image[y+1,x+2] and
                image[y+2,x] and image[y+2,x+1] and image[y+2,x+2]):
                img[y,x] = 255
    return img

def dilation(image):
    """Dilates the image with a 3x3 kernel."""
    width, height = image.shape[1], image.shape[0]
    img = zeros((height, width), dtype=uint8)
    for y in range(height - 2):
        for x in range(width - 2):
            if (image[y,x] or image[y,x+1] or image[y,x+2] or
                image[y+1,x] or image[y+1,x+1] or image[y+1,x+2] or
                image[y+2,x] or image[y+2,x+1] or image[y+2,x+2]):
                img[y,x] = 255
    return img

def measure(image):
    """Locates the boundary box of the shape in the image."""
    size = 0
    width, height = image.shape[1], image.shape[0]
    # Boundary box
    left, top = width, height
    right, bottom = 0, 0
    for y in range(height):
        for x in range(width):
            if image[y,x]:
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

def moments(image):
    """Calculates the moments of the shape in the image."""
    width, height = image.shape[1], image.shape[0]
    m00, m01, m10, m11, m02, m20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for y in range(height):
        for x in range(width):
            if image[y,x]:
                m00 = m00 + 1
                m10 = m10 + x
                m01 = m01 + y
                m11 = m11 + x * y
                m20 = m20 + x * x
                m02 = m02 + y * y
    return [m00, m01, m10, m11, m02, m20]

def ellipse(image):
    """Calculates the ellipse approximation of the shape in the image."""
    m00, m01, m10, m11, m02, m20 = moments(image)
    x_cm, y_cm, angle, length, width = 0, 0, 0, 0, 0
    if m00 > 10:
        x_cm = m10 / m00
        y_cm = m01 / m00
    
        #-- Rounded -------
        m11 = m11 / 64 * 64
        m20 = m20 / 64 * 64
        m02 = m02 / 64 * 64
        #------------------
        
        u20 = float(m20 / m00 - x_cm * x_cm)
        u02 = float(m02 / m00 - y_cm * y_cm)
        u11 = float(m11 / m00 - x_cm * y_cm)
        
        lmax = (u20 + u02 + sqrt(4 * u11 * u11 + (u20 - u02) * (u20 - u02))) / 2
        lmin = (u20 + u02 - sqrt(4 * u11 * u11 + (u20 - u02) * (u20 - u02))) / 2
        
        angle = arctan((2 * u11) / (u20 - u02)) / 2
        length = 4 * sqrt(lmax)
        width = 4 * sqrt(lmin)
    return [x_cm, y_cm, angle, length, width]

def memory_bin(image, filename):
    """Generates the memory initialization file for the testbench simulation."""
    memory = ''
    width, height = image.shape[1], image.shape[0]
    for y in range(height):
        for x in range(width):
            binary = bin(256 | int(image[y,x]))[-8:]
            memory += binary + '\n'
    file = open(filename, 'w')
    file.write(memory)
    file.close()



if __name__ == '__main__':
    import cv2
    
    img = cv2.imread('../../data/frame.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    bin = binarize(img, 128)
    ero = erosion(bin)
       
    bin = erosion(erosion(erosion(erosion(erosion(bin)))))
    
    x_cm, y_cm, angle, length, width = ellipse(ero)
    print angle, length, width
           
    left, top, right, bottom = measure(ero)
    print right - left, bottom - top

    figure()
    subplot(221)
    imshow(img, cmap='gray')
    axis('off')
    subplot(222)
    imshow(bin, cmap='gray')
    axis('off')
    subplot(223)
    imshow(ero, cmap='gray')
    axis('off')
    subplot(224)
    imshow(img, cmap='gray')
    axis('off')
    gca().add_patch(Rectangle([left, top], right - left, bottom - top, facecolor='none', edgecolor='green', lw=1.5))
    gca().add_patch(Ellipse([x_cm, y_cm], length, width, angle=rad2deg(angle), facecolor='none', edgecolor='red', lw=2.5))
    show()
    
    
    from mpl_toolkits.mplot3d import Axes3D
    
    DATA = array([
    [-0.807237702464, 0.904373229492, 111.428744443],
    [-0.802470821517, 0.832159465335, 98.572957317],
    [-0.801052795982, 0.744231916692, 86.485869328],
    [-0.802505546206, 0.642324228721, 75.279804677],
    [-0.804158144115, 0.52882485495, 65.112895758],
    [-0.806418040943, 0.405733109371, 56.1627277595],
    [-0.808515314192, 0.275100227689, 48.508994388],
    [-0.809879521648, 0.139140394575, 42.1027499025],
    [-0.810645106092, -7.48279012695e-06, 36.8668106345],
    [-0.810676720161, -0.139773175337, 32.714580273],
    [-0.811308686707, -0.277276065449, 29.5977405865],
    [-0.812331692291, -0.40975978382, 27.6210856615],
    [-0.816075037319, -0.535615685086, 27.2420699235],
    [-0.823691366944, -0.654350489595, 29.1823292975],
    [-0.836688691603, -0.765630198427, 34.2275056775],
    [-0.854984518665, -0.86845932028, 43.029581434],
    [-0.879261949054, -0.961799684483, 55.9594146815],
    [-0.740499820944, 0.901631050387, 97.0261463995],
    [-0.735011699497, 0.82881933383, 84.971061395],
    [-0.733021568161, 0.740454485354, 73.733621269],
    [-0.732821755233, 0.638770044767, 63.3815970475],
    [-0.733876941678, 0.525818698874, 54.0655910105],
    [-0.735055978521, 0.403303715698, 45.90859502],
    [-0.736448900325, 0.273425879041, 38.935709456],
    [-0.737556181137, 0.13826504904, 33.096106049],
    [-0.738278724065, -9.73058423274e-06, 28.359664343],
    [-0.738507612286, -0.138781586244, 24.627237837],
    [-0.738539663773, -0.275090412979, 21.857410904],
    [-0.739099040189, -0.406068448513, 20.1110519655],
    [-0.741152200369, -0.529726022182, 19.7019157715],
    ])

    Xs = DATA[:,0]
    Ys = DATA[:,1]
    Zs = DATA[:,2]
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    plt.show() # or:
    # fig.savefig('3D.png')
    
