import glob, os
import sys
import numpy as np
import cv2
import dlib

def normalize_point(point, tl, size):
    return np.absolute((point - tl).astype('float')/size.astype('float'))

def get_training_data(image_path, labels_path):
    landmarks_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()

    for dirpath, dirnames, filenames in os.walk(image_path):
        #print dirpath
        files = [name for name in glob.glob(dirpath+"/*.png")]
        if(len(files) != 0):
            img = cv2.imread(max(files),cv2.IMREAD_GRAYSCALE)
            dets = detector(img, 1)
            for k, d in enumerate(dets):
                normalized_landmarks = np.array([[p.x, p.y] for p in landmarks_predictor(img, d).parts()])
                for l in landmarks:
                    cv2.circle(img, (l[0], l[1]), 3, (255,0,0))
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if(len(sys.argv) != 3):
    print "Wrong arguments"
#print sys.argv
get_training_data(sys.argv[1], sys.argv[2])
