import cv2
import numpy as np
from scipy import ndimage

kernel_3x3 = np.array([[-1,-1,-1],
                       [-1,-8,-1],
                       [-1,-1,-1]])

kernel_5x5 = np.array([[-1,-1,-1,-1,-1],
                       [-1, 1, 2, 1,-1],
                       [-1, 2, 4, 2,-1],
                       [-1, 1, 2, 1,-1],
                       [-1,-1,-1,-1,-1]])

img = cv2.imread("./byoda.jpg",0)

k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)

blurred = cv2.GaussianBlur(img,(17,17),0)
g_hpf = img - blurred
g_hpf2 = blurred - img
g_hpf3 = img + blurred

cv2.imshow("3x3", k3)
cv2.imshow("5x5", k5)
cv2.imshow("Blurred", blurred)
cv2.imshow("HPF", g_hpf)
cv2.imshow("HPF2", g_hpf2)
cv2.imshow("HPF3", g_hpf3)
cv2.waitKey(0)
cv2.destroyAllWindows()