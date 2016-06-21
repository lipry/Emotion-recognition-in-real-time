import numpy as np
import cv2
import datetime
import dlib

def normalize_point(point, tl, size):
    return np.absolute((point - tl).astype('float')/size.astype('float'))

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
landmarks_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while(True):
    #t0 = datetime.datetime.now()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rect = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
        landmarks = np.array([[p.x, p.y] for p in landmarks_predictor(gray, rect).parts()])
        print landmarks[0]
        x_rect_min = np.array(landmarks[np.argmin(landmarks[:, 0]),0])
        x_rect_max = np.array(landmarks[np.argmax(landmarks[:, 0]),0])
        y_rect_min = np.array(landmarks[np.argmin(landmarks[:, 1]),1])
        y_rect_max = np.array(landmarks[np.argmax(landmarks[:, 1]),1])
        #print "x min: ", x_rect_min, "x max: ", x_rect_max, "y min", y_rect_min, "y max", y_rect_max
        #cv2.circle(frame, (x_rect_min, y_rect_min), 3, (0,0,255))
        #cv2.circle(frame, (x_rect_min, y_rect_max), 3, (0,0,255))
        #cv2.circle(frame, (x_rect_max, y_rect_max), 3, (0,0,255))
        #cv2.circle(frame, (x_rect_max, y_rect_min), 3, (0,0,255))
        tl_rect = np.array([x_rect_min, y_rect_max])
        normalized_landmarks = np.apply_along_axis(normalize_point, 1, landmarks, tl_rect, np.array([x_rect_max - x_rect_min, y_rect_max - y_rect_min]))
        print normalized_landmarks
        #cv2.circle(frame, (tl_rect[0], tl_rect[1]), 3, (255,0,0))
        #for p in landmarks_predictor(gray, rect).parts():
            #cv2.circle(frame, (p.x, p.y), 3, (255,0,0))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #print "time: ", (datetime.datetime.now() - t0)

cap.release()
cv2.destroyAllWindows()
