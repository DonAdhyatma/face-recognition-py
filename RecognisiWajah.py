import cv2
import os
import numpy as np
wajahDir = 'faces'
latihDir = 'latihwajah'
cam = cv2.VideoCapture(0)
cam.set(3, 650)  # ubah lebar cam
cam.set(4, 650)  # ubah tinggi cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read(latihDir+'/training.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak Diketahui', 'Danni', 'Nama Lain']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1)  # vertical flip
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(
        abuAbu, 1.2, 5, minSize=(round(minWidth), round(minHeight)),)
    # frame, scaleFactor, minNeighbor
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = faceRecognizer.predict(abuAbu[y:y+h, x:x+w])
        # confidence = 0 artinya cocok sempurna
        if confidence <= 50:
            nameID = names[0]
            confidenceTxt = "{0}%".format(round(100-confidence))
        else:
            nameID = names[0]
            confidenceTxt = "{0}%".format(round(100-confidence))
            cv2.putText(frame, str(nameID), (x+5, y-5),
                        font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidenceTxt),
                        (x+5, y+h-5), font, 1, (255, 255, 0), 1)
    cv2.imshow('Recognisi Wajah', frame)
    # cv2.imshow('Webcam - Grey', abuAbu)
    # cv2.imshow('PythonCam 2', abuAbu)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

    if cv2.getWindowProperty('Recognisi Wajah', cv2.WND_PROP_VISIBLE) < 1:
        break

print("EXIT")
cam.release()
cv2.destroyAllWindows()
