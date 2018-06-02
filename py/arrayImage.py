import numpy as np
import skimage.io
# from ransac import *
# from plane_fitting import *
import random
from matplotlib import pylab
from mpl_toolkits import mplot3d

filename = 'C:/Users/Ashley/Documents/GitHub/image-rgb-in-3D/gradientbox.png'
img = skimage.io.imread( filename )

height, width, depth = img.shape
# print(img.shape)
# np.meshgrid( width, height )
# print(np.meshgrid(width, height))
# imageArray = np.append( np.meshgrid( width, height ), img )
# print(imageArray)


heightAr = list(range(0, height))
# print(heightAr)
widthAr = list(range(0, width))
# print(widthAr)
meshGrid = np.meshgrid(widthAr, heightAr)
imageArray = np.concatenate((meshGrid[0][:, :, None], meshGrid[1][:, :, None], img), axis = 2)
# print(imageArray[0, 2])
# print(imageArray.shape)

# reshaped is the input for the plane-fitting
reshaped = imageArray.reshape(width * height, -1)
print(reshaped)


def augment(xyzs):
	axyz = np.ones((len(xyzs), 6))
	# axyz = np.ones((len(xyzs), 4))
	axyz[:, :6] = xyzs
	# axyz[:, :3] = xyzs
	print("augment: ", axyz)
	return axyz

def estimate(xyzs):
	axyz = augment(xyzs[:3])
	print("estimate: ", np.linalg.svd(axyz)[-1][-1, :])
	return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
	print("is_inlier:", np.abs(coeffs.dot(augment([xyz]).T)) < threshold)
	return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

fig = pylab.figure()
ax = mplot3d.Axes3D(fig)

def plot_plane(a, b, c, d):
	xx, yy = np.mgrid[:10, :10]
	print("plot plane: ", xx, yy, (-d - a * xx - b * yy) / c)
	return xx, yy, (-d - a * xx - b * yy) / c

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
	best_ic = 0
	best_model = None
	random.seed(random_seed)
	for i in range(max_iterations):
		s = random.sample(list(data), int(sample_size))
		m = estimate(s)
		ic = 0
		for j in range(len(data)):
			if is_inlier(m, data[j]):
				ic += 1

		print(s)
		print('estimate:', m)
		print('# inliers:', ic)

		if ic > best_ic:
			best_ic = ic
			best_model = m
			if ic > goal_inliers and stop_at_goal:
				break
	print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
	return best_model, best_ic

n = 100
max_iterations = 100
goal_inliers = n * 0.3

# test data
# xyzs = np.random.random((n, 3)) * 10
# xyzs[:50, 2:] = xyzs[:50, :1]

# ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])

# RANSAC
m, b = run_ransac(reshaped, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
a, b, c, d = m
xx, yy, zz = plot_plane(a, b, c, d)
ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))

# print(reshaped[0])

# def printImageArray(imageArray):
#     length = len(imageArray) / 5
#     n = 0
#     for i in range(0, int(length)):
#         print('Pixel #', i, imageArray[n:n+5])
#         n += 5

# printImageArray(imageArray)