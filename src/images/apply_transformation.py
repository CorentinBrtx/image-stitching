from typing import List, Tuple

import cv2
import numpy as np

from src.images.image import Image


def apply_homography(H, point):
    new_point = H @ point
    return new_point[0:2] / new_point[2]


def apply_homography_list(H: np.ndarray, points: List):
    return [apply_homography(H, point) for point in points]


def get_new_corners(image: np.ndarray, H: np.ndarray) -> List[np.ndarray]:
    top_left = np.asarray([[0, 0, 1]]).T
    top_right = np.asarray([[image.shape[1], 0, 1]]).T
    bottom_left = np.asarray([[0, image.shape[0], 1]]).T
    bottom_right = np.asarray([[image.shape[1], image.shape[0], 1]]).T

    return apply_homography_list(H, [top_left, top_right, bottom_left, bottom_right])


def get_offset(corners):
    top_left, top_right, bottom_left = corners[:3]
    return np.array(
        [
            [1, 0, max(0, -float(min(top_left[0], bottom_left[0])))],
            [0, 1, max(0, -float(min(top_left[1], top_right[1])))],
            [0, 0, 1],
        ],
        np.float32,
    )


def get_new_size(corners_images) -> Tuple[int]:
    top_right_x = np.max([corners_image[1][0] for corners_image in corners_images])
    bottom_right_x = np.max([corners_images[3][0] for corners_images in corners_images])

    bottom_left_y = np.max([corners_images[2][1] for corners_images in corners_images])
    bottom_right_y = np.max([corners_images[3][1] for corners_images in corners_images])

    width = int(np.ceil(max(bottom_right_x, top_right_x)))
    height = int(np.ceil(max(bottom_right_y, bottom_left_y)))

    return width, height


def apply_transformation(
    panorama: np.ndarray, image: Image, offset: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    H = offset @ image.H
    corners = get_new_corners(image.image, H)
    added_offset = get_offset(corners)

    corners_image = get_new_corners(image.image, added_offset @ H)
    corners_panorama = get_new_corners(panorama, added_offset)

    size = get_new_size([corners_image, corners_panorama])

    if panorama.shape[0] == 0:
        panorama = np.zeros((*size, 3), dtype=np.uint8)

    panorama = cv2.warpPerspective(panorama, added_offset, size)
    new_image = cv2.warpPerspective(image.image, added_offset @ H, size)

    panorama = np.where(
        np.repeat(np.sum(new_image, axis=2)[:, :, np.newaxis], 3, axis=2) > 0, new_image, panorama
    )

    return panorama, added_offset
