import numpy as np

from src.images import Image
from src.matching.pair_match import PairMatch


def build_homographies(
    connected_components: list[list[Image]], pair_matches: list[PairMatch]
) -> None:
    """
    Build homographies for each image of each connected component, using the pair matches.
    The homographies are saved in the images themselves.

    Args:
        connected_components: The connected components of the panorama
        pair_matches: The valid pair matches
    """
    for connected_component in connected_components:
        component_matches = [
            pair_match for pair_match in pair_matches if pair_match.image_a in connected_component
        ]

        images_added = set()
        current_homography = np.eye(3)

        pair_match = component_matches[0]
        pair_match.compute_homography()

        nb_pairs = len(pair_matches)

        if sum(
            [
                10 * (nb_pairs - i)
                for i, match in enumerate(pair_matches)
                if match.contains(pair_match.image_a)
            ]
        ) > sum(
            [
                10 * (nb_pairs - i)
                for i, match in enumerate(pair_matches)
                if match.contains(pair_match.image_b)
            ]
        ):
            pair_match.image_a.H = np.eye(3)
            pair_match.image_b.H = pair_match.H
        else:
            pair_match.image_b.H = np.eye(3)
            pair_match.image_a.H = np.linalg.inv(pair_match.H)

        images_added.add(pair_match.image_a)
        images_added.add(pair_match.image_b)

        while len(images_added) < len(connected_component):
            for pair_match in component_matches:

                if pair_match.image_a in images_added and pair_match.image_b not in images_added:
                    pair_match.compute_homography()
                    homography = pair_match.H @ current_homography
                    pair_match.image_b.H = pair_match.image_a.H @ homography
                    images_added.add(pair_match.image_b)
                    break

                if pair_match.image_a not in images_added and pair_match.image_b in images_added:
                    pair_match.compute_homography()
                    homography = np.linalg.inv(pair_match.H) @ current_homography
                    pair_match.image_a.H = pair_match.image_b.H @ homography
                    images_added.add(pair_match.image_a)
                    break
