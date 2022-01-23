from typing import List

import cv2
from src.images.image import Image
from src.matches.pair_match import PairMatch


class MultiImageMatches:
    def __init__(self, images: List[Image], ratio: float = 0.75):
        self.images = images
        self.matches = {image.path: {} for image in images}
        self.ratio = ratio

    def get_matches(self, image_a: Image, image_b: Image) -> List:
        if image_b.path not in self.matches[image_a.path]:
            matches = self.compute_matches(image_a, image_b)
            self.matches[image_a.path][image_b.path] = matches

        return self.matches[image_a.path][image_b.path]

    def get_pair_matches(self, max_images=6) -> List[PairMatch]:
        pair_matches = []
        for i, image_a in enumerate(self.images):
            possible_matches = sorted(
                self.images[:i] + self.images[i + 1 :],
                key=lambda image: len(self.get_matches(image_a, image)),
                reverse=True,
            )[:max_images]
            for image_b in possible_matches:
                if self.images.index(image_b) > i:
                    pair_match = PairMatch(image_a, image_b, self.get_matches(image_a, image_b))
                    if pair_match.is_valid():
                        pair_matches.append(pair_match)
        return pair_matches

    def compute_matches(self, image_a, image_b):

        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches = []

        rawMatches = matcher.knnMatch(image_a.features, image_b.features, 2)
        matches = []

        for m, n in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < n.distance * self.ratio:
                matches.append(m)

        return matches
