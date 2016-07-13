import numpy as np
import dlib

class faceLandmarks:
    def __init__(self, dataset_path):
        self.predictor = dlib.shape_predictor(dataset_path)

    def normalize_point(self, point, tl, size):
        return np.absolute((point - tl).astype('float')/size.astype('float'))

    def get_landmarks(self, image, rectangle):
        landmarks = np.array([[p.x, p.y] for p in self.predictor(image, rectangle).parts()])
        tl = np.array([rectangle.left(), rectangle.top()])
        size = np.array([rectangle.width(), rectangle.height()])
        normalized_landmarks = np.apply_along_axis(self.normalize_point, 1, landmarks, tl, size)
        return np.hstack(normalized_landmarks)
