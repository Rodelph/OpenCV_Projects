import os
import cv2

Name = input("Please enter your full name : ")
output_folder = './recog_Rsc/' + Name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

camera = cv2.VideoCapture(0)
count = 0
i = 0

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