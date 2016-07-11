import glob, os
import sys
import numpy as np
import cv2
import dlib
import face_landmarks
import pickle
import datasets

def get_training_data(image_path, labels_path):
    landmarks_predictor = face_landmarks.faceLandmarks("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()

    features = []
    labels = []
    for dirpath, dirnames, filenames in os.walk(image_path):
        files = [name for name in glob.glob(dirpath+"/*.png")]

        if(len(files) != 0): #if images exist
            file_path = max(files)
            pure_name = os.path.splitext(os.path.basename(file_path))[0]
            emotion = glob.glob(labels_path+"/*/*/"+pure_name+"_emotion.txt")
            if(len(emotion) > 0): #data about emotion exist
                in_file = open(emotion[0],"r")
                label = int(float(in_file.read()))
                in_file.close()
                labels.append(label)
                img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                dets = detector(img, 1)
                #in base case will be iterate only one time (every image in CK+ dataset have one face)
                for k, rect in enumerate(dets):
                    landmarks = landmarks_predictor.get_landmarks(img, rect)
                    features.append(landmarks)
    return features, labels

if(len(sys.argv) != 3):
    print "Wrong arguments"


print "Extracting training data from dataset..."
features, labels = get_training_data(sys.argv[1], sys.argv[2])

print "Saving features and labels in file"
with open(datasets.EMOTION_FEATURES_DATASET, "w") as outfile:
    pickle.dump([features, labels], outfile)
