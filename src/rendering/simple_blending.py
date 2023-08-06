import cv2
import numpy as np

from src.images import Image
from src.rendering.utils import get_new_parameters, single_weights_matrix


def add_image(
    panorama: np.ndarray, image: Image, offset: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add a new image to the panorama using the provided offset and weights.

    Args:
        panorama: Existing panorama
        image: Image to add to the panorama
        offset: Offset already applied to the panorama
        weights: Weights matrix of the panorama

    Returns:
        panorama: Panorama with the new image
        offset: New offset matrix
        weights: New weights matrix
    """
    H = offset @ image.H
    size, added_offset = get_new_parameters(panorama, image.image, H)

    new_image = cv2.warpPerspective(image.image, added_offset @ H, size)

    if panorama is None:
        panorama = np.zeros_like(new_image)
        weights = np.zeros_like(new_image)
    else:
        panorama = cv2.warpPerspective(panorama, added_offset, size)
        weights = cv2.warpPerspective(weights, added_offset, size)

    image_weights = single_weights_matrix(image.image.shape)
    image_weights = np.repeat(
        cv2.warpPerspective(image_weights, added_offset @ H, size)[:, :, np.newaxis], 3, axis=2
    )

    normalized_weights = np.zeros_like(weights)
    normalized_weights = np.divide(
        weights, (weights + image_weights), where=weights + image_weights != 0
    )

    panorama = np.where(
        np.logical_and(
            np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
            np.repeat(np.sum(new_image, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
        ),
        0,
        new_image * (1 - normalized_weights) + panorama * normalized_weights,
    ).astype(np.uint8)

    new_weights = (weights + image_weights) / (weights + image_weights).max()

    return panorama, added_offset @ offset, new_weights


def simple_blending(images: list[Image]) -> np.ndarray:
    """
    Build a panorama from the given images using simple blending.

    Args:
        images: Images to build the panorama from

    Returns:
        panorama: Panorama of the given images
    """
    panorama = None
    weights = None
    offset = np.eye(3)
    for image in images:
        panorama, offset, weights = add_image(panorama, image, offset, weights)

    return panorama
