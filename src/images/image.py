import cv2
import numpy as np


class Image:
    def __init__(self, path: str):
        self.path = path
        self.image = cv2.imread(path)
        self.keypoints = None
        self.features = None
        self.H = np.eye(3)
        self.component = 0
        self.gain = np.ones(3, dtype=np.float32)

    def compute_features(self):
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(self.image, None)
        self.keypoints = keypoints
        self.features = features

    def set_homography(self, H: np.ndarray):
        self.H = H
