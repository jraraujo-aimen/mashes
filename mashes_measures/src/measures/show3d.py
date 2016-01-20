import numpy as np
from mayavi import mlab

X = np.array([0, 1, 0, 1, 0.75])
Y = np.array([0, 0, 1, 1, 0.75])
Z = np.array([1, 1, 1, 1, 2])

# Define the points in 3D space
# including color code based on Z coordinate.
pts = mlab.points3d(X, Y, Z, Z)

# Triangulate based on X, Y with Delaunay 2D algorithm.
# Save resulting triangulation.
mesh = mlab.pipeline.delaunay2d(pts)

# Remove the point representation from the plot
pts.remove()

# Draw a surface based on the triangulation
surf = mlab.pipeline.surface(mesh)

# Simple plot.
mlab.xlabel("x")
mlab.ylabel("y")
mlab.zlabel("z")
mlab.show()



"""Test surf on regularly spaced co-ordinates like MayaVi."""
def f(x, y):
    sin, cos = numpy.sin, numpy.cos
    return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

x, y = numpy.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
s = mlab.surf(x, y, f)
#cs = contour_surf(x, y, f, contour_z=0)
mlab.show()
