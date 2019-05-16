from skimage import io
from skimage.color import rgb2gray
from scipy import signal as sig
import numpy as np


img = io.imread('2.png')
gray = rgb2gray(img)
io.imshow(gray)
io.show()
def grad_x(gray):
	kernel_x = np.array([[-1, 0, +1],
						 [-2, 0, +2],
						 [-1, 0, +1]])
	return sig.convolve2d(gray, kernel_x, mode='same')

def grad_y(gray):
	kernel_y = np.array([[+1, +2, +1],
						 [ 0,  0,  0],
						 [-1, -2, -1]])
	return sig.convolve2d(gray, kernel_y, mode='same')

I_x = grad_x(gray)
I_y = grad_y(gray)

Ixx = I_x**2
Ixy = I_x*I_y
Iyy = I_y**2

corner = np.zeros(gray.shape)
offset = 1
k = 0.04
print(gray.shape)
height = gray.shape[0]
width = gray.shape[1]
for y in range(offset, height-offset):
	for x in range(offset, width-offset):
		Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
		Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
		Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
		det = (Sxx * Syy) - Sxy**2
		trace = Sxx + Syy
		r = det - k*(trace**2)
		if r > 0:
			corner[y,x] = 255
io.imshow(corner)
io.show()