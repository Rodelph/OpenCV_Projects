import cv2
import numpy as np

imgT = cv2.pyrDown(cv2.imread("eye.jpg", cv2.IMREAD_UNCHANGED))
ret, thresh = cv2.threshold(cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#for a normal contour detection
for c in contours:
    # find bounding box coordinates
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(imgT, (x,y), (x+w, y+h), (0, 255, 0), 2)

    # find minimum area
    rect = cv2.minAreaRect(c)
    # calculate coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    # normalize coordinates to integers
    box = np.int0(box)
    # draw contours
    cv2.drawContours(imgT, [box], 0, (0,0, 255), 3)

    # calculate center and radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    # cast to integers
    center = (int(x), int(y))
    radius = int(radius)
    # draw the circle
    imgT = cv2.circle(imgT, center, radius, (0, 255, 0), 2)

#for convex contour detection using Douglas-Peucker algorithm

black = np.zeros_like(imgT)
for cnt in contours : 
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon ,True)
    hull = cv2.convexHull(cnt)
    cv2.drawContours(black, [cnt], -1, (0,255,0), 2)
    cv2.drawContours(black, [approx], -1, (255,255,0), 2)
    cv2.drawContours(black, [hull], -1, (255,255,0), 2)

cv2.drawContours(imgT, contours,-1, (255,0,0), 1)
cv2.imshow("contours", imgT)
cv2.imshow("hull",black)

cv2.waitKey()
cv2.destroyAllWindows()