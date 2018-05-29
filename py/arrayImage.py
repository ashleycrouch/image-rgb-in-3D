# from numpy import *
import numpy as np
# np.set_printoptions( threshold=np.nan )
import skimage.io

filename = 'C:/Users/Ashley/Documents/GitHub/image-rgb-in-3D/gradientbox.png'
img = skimage.io.imread( filename )

height, width, depth = img.shape
print(img.shape)
# np.meshgrid( width, height )
# print(np.meshgrid(width, height))
# imageArray = np.append( np.meshgrid( width, height ), img )
# print(imageArray)


heightAr = list(range(0, height))
# print(heightAr)
widthAr = list(range(0, width))
# print(widthAr)
# print(np.cross(heightAr, widthAr))
meshGrid = np.meshgrid(widthAr, heightAr)
imageArray = np.concatenate((meshGrid[0][:, :, None], meshGrid[1][:, :, None], img), axis = 2)
print(imageArray[0, 2])
# print(imageArray.shape)

# reshaped is the input for the plane-fitting
reshaped = imageArray.reshape(width * height, -1)
print(reshaped)
print(reshaped[0])

def printImageArray(imageArray):
    length = len(imageArray) / 5
    n = 0
    for i in range(0, int(length)):
        print('Pixel #', i, imageArray[n:n+5])
        n += 5

# printImageArray(imageArray)