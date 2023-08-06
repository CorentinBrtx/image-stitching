import cv2
import numpy as np

from src.images import Image
from src.rendering.utils import get_new_parameters, single_weights_matrix


def add_weights(
    weights_matrix: np.ndarray, image: Image, offset: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add the weights corresponding to the given image to the given existing weights matrix.

    Args:
        weights_matrix: Existing weights matrix
        image: New image to add to the weights matrix
        offset: Offset already applied to the weights matrix

    Returns:
        weights_matrix, offset: The updated weights matrix and the updated offset
    """
    H = offset @ image.H
    size, added_offset = get_new_parameters(weights_matrix, image.image, H)

    weights = single_weights_matrix(image.image.shape)
    weights = cv2.warpPerspective(weights, added_offset @ H, size)[:, :, np.newaxis]

    if weights_matrix is None:
        weights_matrix = weights
    else:
        weights_matrix = cv2.warpPerspective(weights_matrix, added_offset, size)

        if len(weights_matrix.shape) == 2:
            weights_matrix = weights_matrix[:, :, np.newaxis]

        weights_matrix = np.concatenate([weights_matrix, weights], axis=2)

    return weights_matrix, added_offset @ offset


def get_max_weights_matrix(images: list[Image]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the maximum weights matrix for the given images.

    Args:
        images: List of images to compute the maximum weights matrix for

    Returns:
        max_weights_matrix: Maximum weights matrix
        offset: Offset matrix
    """
    weights_matrix = None
    offset = np.eye(3)

    for image in images:
        weights_matrix, offset = add_weights(weights_matrix, image, offset)

    weights_maxes = np.max(weights_matrix, axis=2)[:, :, np.newaxis]
    max_weights_matrix = np.where(
        np.logical_and(weights_matrix == weights_maxes, weights_matrix > 0), 1.0, 0.0
    )

    max_weights_matrix = np.transpose(max_weights_matrix, (2, 0, 1))

    return max_weights_matrix, offset


def get_cropped_weights(
    images: list[Image], weights: np.ndarray, offset: np.ndarray
) -> list[np.ndarray]:
    """
    Convert a global weights matrix to a list of weights matrices for each image,
    where each weight matrix is the size of the corresponding image.

    Args:
        images: List of images to convert the weights matrix for
        weights: Global weights matrix
        offset: Offset matrix

    Returns:
        cropped_weights: List of weights matrices for each image
    """
    cropped_weights = []
    for i, image in enumerate(images):
        cropped_weights.append(
            cv2.warpPerspective(
                weights[i], np.linalg.inv(offset @ image.H), image.image.shape[:2][::-1]
            )
        )

    return cropped_weights


def build_band_panorama(
    images: list[Image],
    weights: list[np.ndarray],
    bands: list[np.ndarray],
    offset: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """
    Build a panorama from the given bands and weights matrices.
    The images are needed for their homographies.

    Args:
        images: Images to build the panorama from
        weights: Weights matrices for each image
        bands: Bands for each image
        offset: Offset matrix
        size: Size of the panorama

    Returns:
        panorama: Panorama for the given bands and weights
    """
    pano_weights = np.zeros(size)
    pano_bands = np.zeros((*size, 3))

    for i, image in enumerate(images):
        weights_at_scale = cv2.warpPerspective(weights[i], offset @ image.H, size[::-1])
        pano_weights += weights_at_scale
        pano_bands += weights_at_scale[:, :, np.newaxis] * cv2.warpPerspective(
            bands[i], offset @ image.H, size[::-1]
        )

    return np.divide(
        pano_bands, pano_weights[:, :, np.newaxis], where=pano_weights[:, :, np.newaxis] != 0
    )


def multi_band_blending(images: list[Image], num_bands: int, sigma: float) -> np.ndarray:
    """
    Build a panorama from the given images using multi-band blending.

    Args:
        images: Images to build the panorama from
        num_bands: Number of bands to use for multi-band blending
        sigma: Standard deviation for the multi-band blending

    Returns:
        panorama: Panorama after multi-band blending
    """
    max_weights_matrix, offset = get_max_weights_matrix(images)
    size = max_weights_matrix.shape[1:]

    max_weights = get_cropped_weights(images, max_weights_matrix, offset)

    weights = [[cv2.GaussianBlur(max_weights[i], (0, 0), 2 * sigma) for i in range(len(images))]]
    sigma_images = [cv2.GaussianBlur(image.image, (0, 0), sigma) for image in images]
    bands = [
        [
            np.where(
                images[i].image.astype(np.int64) - sigma_images[i].astype(np.int64) > 0,
                images[i].image - sigma_images[i],
                0,
            )
            for i in range(len(images))
        ]
    ]

    for k in range(1, num_bands - 1):
        sigma_k = np.sqrt(2 * k + 1) * sigma
        weights.append(
            [cv2.GaussianBlur(weights[-1][i], (0, 0), sigma_k) for i in range(len(images))]
        )

        old_sigma_images = sigma_images

        sigma_images = [
            cv2.GaussianBlur(old_sigma_image, (0, 0), sigma_k)
            for old_sigma_image in old_sigma_images
        ]
        bands.append(
            [
                np.where(
                    old_sigma_images[i].astype(np.int64) - sigma_images[i].astype(np.int64) > 0,
                    old_sigma_images[i] - sigma_images[i],
                    0,
                )
                for i in range(len(images))
            ]
        )

    weights.append([cv2.GaussianBlur(weights[-1][i], (0, 0), sigma_k) for i in range(len(images))])
    bands.append([sigma_images[i] for i in range(len(images))])

    panorama = np.zeros((*max_weights_matrix.shape[1:], 3))

    for k in range(0, num_bands):
        panorama += build_band_panorama(images, weights[k], bands[k], offset, size)
        panorama[panorama < 0] = 0
        panorama[panorama > 255] = 255

    return panorama
