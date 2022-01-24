import argparse
import glob
import logging
import os
import time
from typing import List

import cv2
import numpy as np

from src.images import Image, apply_transformation, get_gain_compensations
from src.matches import MultiImageMatches, PairMatch, find_connected_components

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(dest="data_dir", help="Directory containing the images")
args = vars(parser.parse_args())

image_paths = glob.glob(os.path.join(args["data_dir"], "*.jpg"))

images = [Image(path) for path in image_paths]

for image in images:
    image.compute_features()

matcher = MultiImageMatches(images)
pair_matches: List[PairMatch] = matcher.get_pair_matches()
pair_matches.sort(key=lambda pair_match: len(pair_match.matches), reverse=True)

connected_components = find_connected_components(pair_matches)

for connected_component in connected_components:

    component_matches = [
        pair_match for pair_match in pair_matches if pair_match.image_a in connected_component
    ]

    images_added = set()
    current_homography = np.eye(3)

    pair_match = component_matches[0]
    pair_match.compute_homography()
    pair_match.image_a.set_homography(np.eye(3))
    pair_match.image_b.set_homography(pair_match.H)
    images_added.add(pair_match.image_a)
    images_added.add(pair_match.image_b)

    while len(images_added) < len(connected_component):
        for pair_match in component_matches:

            if pair_match.image_a in images_added and pair_match.image_b not in images_added:
                pair_match.compute_homography()
                homography = pair_match.H @ current_homography
                pair_match.image_b.set_homography(pair_match.image_a.H @ homography)
                images_added.add(pair_match.image_b)
                break

            elif pair_match.image_a not in images_added and pair_match.image_b in images_added:
                pair_match.compute_homography()
                homography = np.linalg.inv(pair_match.H) @ current_homography
                pair_match.image_a.set_homography(pair_match.image_b.H @ homography)
                images_added.add(pair_match.image_a)
                break

time.sleep(0.1)

for connected_component in connected_components:
    component_matches = [
        pair_match for pair_match in pair_matches if pair_match.image_a in connected_components[0]
    ]

    gains = get_gain_compensations(connected_components[0], component_matches)

    for i, image in enumerate(connected_components[0]):
        image.gain = gains[i]

time.sleep(0.1)

results = []

for connected_component in connected_components:
    result = np.zeros((0, 0, 3))
    offset = np.eye(3)
    for image in connected_component:
        result, added_offset = apply_transformation(result, image, offset)
        offset = added_offset @ offset
    results.append(result)


os.makedirs(os.path.join(args["data_dir"], "results"), exist_ok=True)
for i, result in enumerate(results):
    cv2.imwrite(os.path.join(args["data_dir"], "results", f"pano_{i}.jpg"), result)
