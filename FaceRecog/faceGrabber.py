import os
import cv2

# Insert a Name for the person so that he can be identified by the program in the terminal.

Name = input("Please enter your full name : ")

# The inserted name is going to be used to created a folder where the images are going to be stored.
  
output_folder = './recog_Rsc/' + Name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Creating a variables using the CascadeClassifier method from OpenCV, and using the haarcascade_frontalface xml, as well as 
# the haarcascade eye xml files.

face_cascade = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./xml/haarcascade_eye.xml')

# Initializing the camera variable that holds the index of our webcam, as well as two integers for counting purposes.

camera = cv2.VideoCapture(0)
count = 0
i = 0

# Creating a loop where we are going to use all the variable that we already declared in order to store the image frame that are being captured
# by the webcam. 

while(cv2.waitKey(1) == -1):
    success, frame = camera.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize = (120, 120))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x + w, y + h), (255,0,0), 2)
            face_img = cv2.resize(gray[y:y+h , x:x+w], (200,200))
            face_filename = '%s/%d.pgm' % (output_folder, count)
            cv2.imwrite(face_filename, face_img)
            terminal_output = ("Image N°%d has been captured!")%(i)
            print(terminal_output)
            i += 1
            count += 1
        cv2.imshow("Capturing faces ...", frame)