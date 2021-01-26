import cv2
import numpy as np


img = cv2.imread("eye.jpg")
img = cv2.resize(img, (720, 480),interpolation= cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 120)
minLineLength = 20
maxLineGap = 5

lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, 20, minLineLength, maxLineGap)
for x1, y1, x2, y2  in lines[0]:
    cv2.line(img, (x1,y1),(x2,y2), (0,255,0),2)

cv2.imshow("edges", edges)
cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows()


img1 = cv2.imread("eye.jpg")
img1 = cv2.resize(img, (720, 480),interpolation= cv2.INTER_AREA)
gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img = cv2.medianBlur(gray_img, 5)
circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 120, param1= 100, param2= 30, minRadius= 0, maxRadius= 0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    #draw outer circle
    cv2.circle(img1, (i[0],i[1]),i[2], (0,255,0), 2)
    #draw the center of the circle 
    cv2.circle(img1, (i[0],i[1]), 2, (0,0,255), 3)

cv2.imwrite("Eye_Circle_Detection.jpg", img1)
cv2.imshow("HoughCircles", img1)
cv2.waitKey()
cv2.destroyAllWindows()