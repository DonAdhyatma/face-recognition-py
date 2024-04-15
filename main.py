import cv2
cam = cv2.VideoCapture(0)
# implementasi algorita haar cascade
while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('PythonCam', frame)
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
