# implementasi algoritma haar cascade
# deteksi wajah: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
# deteksi mata: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
# langkah untuk face recognition: rekam data wajah, training data wajah, recognition
import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 650)  # ubah lebar cam
cam.set(4, 650)  # ubah tinggi cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5)
    # frame, scaleFactor, minNeighbor
    for (x, y, w, h) in faces:
        # susunan warna rgb di python dibaca terbalik jadi mulai dari bgr, blue green, red
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('PythonCam', frame)
    # cv2.imshow('Webcam - Grey', abuAbu)
    cv2.imshow('PythonCam 2', abuAbu)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('q'):
        break

    if cv2.getWindowProperty('PythonCam', cv2.WND_PROP_VISIBLE) < 1:
        break

cam.release()
cv2.destroyAllWindows()
