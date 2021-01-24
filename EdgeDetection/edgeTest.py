import cv2
import numpy as np

img = cv2.imread("./byoda.jpg", 0)
#Edge detection with Canny function. The canny edge detection algorithm is complex. It is a five step process : 
# - Denoise the image with a Gaussian filter
# - Calculate the gradients 
# - Apply non maximum suppression (NMS) on the edges. Basically, this means that the algorithm selects the best edges from a set of overlapping edges.
# - Apply a double threshol to all the detected edges to eliminate any false positive 
# - Analyse all the edges and their connection to each other to keep the real edges and discard the weak ones
cv2.imwrite("canny.jpg", cv2.Canny(img, 200,300))
cv2.imshow("Canny", cv2.imread("./canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()