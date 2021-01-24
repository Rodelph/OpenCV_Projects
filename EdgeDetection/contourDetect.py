import cv2
import numpy as np 

"""
We first create an empty black image that is 200 x 200 pixels in size. Then, we place a white square in the center of it by utilizing array's ability to assign values on 
a slice. Then we threshold the image and call the findContours function. This function has three parameters : 
 - The input image 
 - The hierarchy type
 - The contour approxiamtion method
"""

img = np.zeros((200,200), dtype=np.uint8)
img[50:50, 50:150] = 255
ret, thresh = cv2.threshold(img, 127,255,0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #One of the supported values is cv2.RETR_TREE, which tells the function 
                                                                                       #to retrieve the entire hierarchy of external and internal contours.
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color,contours, -1, (0,255,0),2)
cv2.imshow("Contours", color)
cv2.waitKey()
cv2.destroyAllWindows()