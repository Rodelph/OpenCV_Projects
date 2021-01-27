import numpy as np 
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("./res/koh2.jpg")
imgT = cv2.imread("./res/koh2.jpg")
nImg = cv2.resize(imgT, (720,480), interpolation=  cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# After converting the image from bgr to grayscale, we runa threshold on it. This operation help by dividing the image into two regions, blacks and whites.

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# We remove the noise from the thresholded image by applying a morphological transformation to it. Morphology consists of dilating (expanding) or reoding (contracting)
# the white regions of the image in soem series of steps. We will apply the morphological open operation whiuch consists of a erosion step followed by a dilation sterp.
# The open operation makes big white regions swallow up little black regions (noise), while leaving big black regions (real objects) relatively unchanged. 

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

# By dilating the result ofthe open transformation, we can obtain regions of the image that are msot certainly background.

sure_bg = cv2.dilate(opening, kernel, iterations= 3)

# Once we have obtained the distanceTransform representation of the image, we apply a threshold to select regtions that are most surely aprt of the foreground

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)

# At this stage we have some sure forground and background regions. As for the in between, we can find theses unknown regions by substracting the sure foreground 
# from background.

unknown = cv2.subtract(sure_bg, sure_fg)

# NOw that we have these regions, we can build our barriers. This is done with the connectedComponents function. We took a glimpse at graph theory when we analyzed
# GrabCut algorithm and conceptualized an image as a set of n,odes that are connected by edges. Given the sure foreground areas, some of these nodes will be connected
# by edges. Given the sure foreground areas some oif these nodes will be connected together, buyt some will not. The disconnected nodes belong to different water 
# valleys, and there should be a barrier between them.

ret, markers = cv2.connectedComponents(sure_fg)

markers += 1 # Add one to all labels so taht sure background is not 0 but 1
markers[unknown==255] = 0 # Label unknown regions as 0

# FInally, we open the gates! Let the water flow! The cv2.watershed functions assigns the label -1 to pixels that are edges between componenets. We color these edges 
# blue in the original images 

markers = cv2.watershed(img, markers)
img[markers==-1] = [255, 0, 0]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imshow("test", nImg)
cv2.waitKey()