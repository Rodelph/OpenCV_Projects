import os 
import cv2
import numpy as np 

def read_images(path, image_size):
    names = []
    training_images, training_labels = [], []
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        i = 0
        j = 0
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                training_images.append(img)
                terminal_img_output = ("Inserting image number %d has been done !")%(i)
                print(terminal_img_output)
                i += 1                
                training_labels.append(label)
                terminal_lbl_output = ("Inserting label number %d has been done !")%(j)
                print(terminal_lbl_output)
                j += 1
                global resultTrainImg
                resultTrainImg = i
            label += 1
    training_images = np.asarray_chkfinite(training_images, np.uint8)
    training_labels = np.asarray_chkfinite(training_labels, np.int32)
    return names, training_images, training_labels

path_to_training_image = './recog_Rsc'
training_image_size = (200,200)
names, training_images, training_labels = read_images(path_to_training_image,training_image_size)
model = cv2.face.EigenFaceRecognizer_create()
model.train(training_images, training_labels)
face_cascade = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
camera.set(3,1260)
camera.set(4,720)    

while (cv2.waitKey(1) == -1):
    success, frame = camera.read()
    if success:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[x:x+w, y:y+w]
            if roi_gray.size == 0:
                continue
            roi_gray = cv2.resize(roi_gray, training_image_size)
            label, confidence = model.predict(roi_gray)
            text = '%s, confidence = %0.2f' % (names[label], confidence)
            cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 0), 2)
        cv2.imshow("Face recognition", frame)