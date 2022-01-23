import cv2
import numpy as np

from src.images.image import Image


class PairMatch:
    def __init__(self, image_a: Image, image_b: Image, matches=None):
        self.image_a = image_a
        self.image_b = image_b
        self.matches = matches
        self.H = None
        self.status = None
        self.overlap = None
        self.Iab = None
        self.Iba = None
        self.matchpoints_a = None
        self.matchpoints_b = None

    def compute_homography(self, ransac_reproj_thresh: float = 5, ransac_max_iter: int = 500):
        self.matchpoints_a = np.float32(
            [self.image_a.keypoints[match.queryIdx].pt for match in self.matches]
        )
        self.matchpoints_b = np.float32(
            [self.image_b.keypoints[match.trainIdx].pt for match in self.matches]
        )

        self.H, self.status = cv2.findHomography(
            self.matchpoints_b,
            self.matchpoints_a,
            cv2.RANSAC,
            ransac_reproj_thresh,
            maxIters=ransac_max_iter,
        )

    def set_overlap(self):

        if self.H is None:
            self.compute_homography()

        mask_a = np.zeros_like(self.image_a.image[:, :, 0], dtype=np.uint8)
        mask_b = cv2.warpPerspective(
            np.zeros_like(self.image_b.image[:, :, 0], dtype=np.uint8), self.H, mask_a.shape[::-1]
        )

        self.overlap = mask_a * mask_b

    def is_valid(self, alpha: float = 8, beta: float = 0.3):
        if self.overlap is None:
            self.set_overlap()

        if self.status is None:
            self.compute_homography()

        matches_in_overlap = self.matchpoints_a[
            self.overlap[
                self.matchpoints_a[:, 1].astype(np.int64),
                self.matchpoints_a[:, 0].astype(np.int64),
            ]
            == 1
        ]

        return self.status.sum() > alpha + beta * matches_in_overlap.shape[0]

    def is_in(self, image: Image) -> bool:
        return self.image_a == image or self.image_b == image

    def set_intensities(self):
        if self.overlap is None:
            self.set_overlap()

        Ia = self.image_a.sum(axis=2) / 3
        Ib = self.image_b.sum(axis=2) / 3

        inverse_overlap = cv2.warpPerspective(
            self.overlap, np.linalg.inv(self.H), self.image_b.shape[1::-1]
        )

        self.Iab = sum(Ia * self.overlap) / self.overlap.sum()
        self.Iba = sum(Ib * inverse_overlap) / inverse_overlap.sum()
