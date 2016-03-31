# Measures

Measurements: major_axis, minor_axis, orientation.


Call geometry.py

	def find_geometry(frame):
		cnt=melt_pool.find_contour(img_bin)
		if cnt is not None:
			ellipse = melt_pool.find_ellipse(cnt)
			(x, y), (h, v), angle = ellipse
