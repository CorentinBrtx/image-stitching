import cv2
import numpy as np


class Image:
    def __init__(self, path: str):
        """
        Image constructor.

        Parameters
        ----------
        path : str
            path to the image
        """
        self.path = path
        self.image: np.ndarray = cv2.imread(path)
        self.keypoints = None
        self.features = None
        self.H: np.ndarray = np.eye(3)
        self.component_id: int = 0
        self.gain: np.ndarray = np.ones(3, dtype=np.float32)

    def compute_features(self):
        """
        Compute the features and the keypoints of the image using SIFT.
        """
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(self.image, None)
        self.keypoints = keypoints
        self.features = features
