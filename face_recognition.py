import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['MS Dhoni', 'Sachin Tendulkar', 'Virat Kohli']
# features = np.load('features.yml')
# labels = np.load('labels.yml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users\harsh\Desktop\OPENCV\Faces_Dataset\test\Virat Kohli\images.jpg')
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cv.imshow('Person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'{people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0,  (0,255,0), thickness = 2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness = 2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)