import numpy as np
import cv2
import datetime

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("smiled_05.xml")

cap = cv2.VideoCapture(0)

while(True):
    #t0 = datetime.datetime.now()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img_g = gray[y:y+h, x:x+w]
        face_img_c = frame[y:y+h, x:x+w]
        smiles = smileCascade.detectMultiScale(
            face_img_g,
            scaleFactor=2.0,
            minNeighbors=25,
            minSize=(35, 35),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        for (x, y, w, h) in smiles:
            cv2.rectangle(face_img_c, (x, y), (x+w, y+h), (0,0,255), 2)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #print "time: ", (datetime.datetime.now() - t0)

cap.release()
cv2.destroyAllWindows()
