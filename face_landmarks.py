import numpy as np
import dlib

class faceLandmarks:
    def __init__(self, dataset_path):
        self.predictor = dlib.shape_predictor(dataset_path)

    def normalize_point(self, point, tl, size):
        return np.absolute((point - tl).astype('float')/size.astype('float'))

    def get_landmarks(self, image, rectangle):
        landmarks = np.array([[p.x, p.y] for p in self.predictor(image, rectangle).parts()])
        x_rect_min = np.array(landmarks[np.argmin(landmarks[:, 0]),0])
        x_rect_max = np.array(landmarks[np.argmax(landmarks[:, 0]),0])
        y_rect_min = np.array(landmarks[np.argmin(landmarks[:, 1]),1])
        y_rect_max = np.array(landmarks[np.argmax(landmarks[:, 1]),1])
        tl_rect = np.array([x_rect_min, y_rect_max])
        normalized_landmarks = np.apply_along_axis(self.normalize_point, 1, landmarks, tl_rect, np.array([x_rect_max - x_rect_min, y_rect_max - y_rect_min]))
        return np.hstack(normalized_landmarks)
