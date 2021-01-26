import numpy as np 
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("./koh2.jpg")
imgT = cv2.imread("./koh2.jpg")
nImg = cv2.resize(imgT, (720,480), interpolation=  cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#After converting the image from bgr to grayscale, we runa threshold on it. This operation help by dividing the image into two regions, blacks and whites.

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

#We remove the noise from the thresholded image by applying a morphological transformation to it. Morphology consists of dilating (expanding) or reoding (contracting)
#the white regions of the image in soem series of steps. We will apply the morphological open operation whiuch consists of a erosion step followed by a dilation sterp.
#The open operation makes big white regions swallow up little black regions (noise), while leaving big black regions (real objects) relatively unchanged. 

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

sure_bg = cv2.dilate(opening, kernel, iterations= 3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)
unknown = cv2.subtract(sure_bg, sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown==255] = 0
markers = cv2.watershed(img, markers)
img[markers==-1] = [255, 0, 0]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imshow("test", nImg)
cv2.waitKey()