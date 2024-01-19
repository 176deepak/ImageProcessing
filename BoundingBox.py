import cv2 as cv
import numpy as np   

# Load the pre-trained face detection model
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Camera is not accessible! Exiting.....")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't recieve frames, exiting...")
    frame = cv.flip(frame, 180)
    face = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in face:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow("Live Face Capturing", frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()