from typing import List, Tuple

import numpy as np


def apply_homography(H: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Apply a homography to a point.

    Parameters
    ----------
    H : np.ndarray,
        Homography matrix.
    point : np.ndarray,
        Point to apply the homography to, with shape (2,1).

    Returns
    -------
    new_point : np.ndarray,
        Point after applying the homography, with shape (2,1).
    """
    point = np.asarray([[point[0][0], point[1][0], 1]]).T
    new_point = H @ point
    return new_point[0:2] / new_point[2]


def apply_homography_list(H: np.ndarray, points: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply a homography to a list of points.

    Parameters
    ----------
    H : np.ndarray
        Homography matrix.
    points : List
        List of points to apply the homography to, each with shape (2,1).

    Returns
    -------
    new_points : List
        List of points after applying the homography, each with shape (2,1).
    """
    return [apply_homography(H, point) for point in points]


def get_new_corners(image: np.ndarray, H: np.ndarray) -> List[np.ndarray]:
    """
    Get the new corners of an image after applying a homography.

    Parameters
    ----------
    image : np.ndarray
        Original image.
    H : np.ndarray
        Homography matrix.

    Returns
    -------
    corners : List[np.ndarray]
        Corners of the image after applying the homography.
    """
    top_left = np.asarray([[0, 0]]).T
    top_right = np.asarray([[image.shape[1], 0]]).T
    bottom_left = np.asarray([[0, image.shape[0]]]).T
    bottom_right = np.asarray([[image.shape[1], image.shape[0]]]).T

    return apply_homography_list(H, [top_left, top_right, bottom_left, bottom_right])


def get_offset(corners: List[np.ndarray]) -> np.ndarray:
    """
    Get offset matrix so that all corners are in positive coordinates.

    Parameters
    ----------
    corners : List[np.ndarray]
        List of corners of the image.

    Returns
    -------
    offset : np.ndarray
        Offset matrix.
    """
    top_left, top_right, bottom_left = corners[:3]
    return np.array(
        [
            [1, 0, max(0, -float(min(top_left[0], bottom_left[0])))],
            [0, 1, max(0, -float(min(top_left[1], top_right[1])))],
            [0, 0, 1],
        ],
        np.float32,
    )


def get_new_size(corners_images: List[List[np.ndarray]]) -> Tuple[int, int]:
    """
    Get the size of the image that would contain all the given corners

    Parameters
    ----------
    corners_images : List[List[np.ndarray]]
        List of corners of the images (i.e. corners_images[i] is the list of corners of image i).

    Returns
    -------
    (width, height) : Tuple[int, int]
        Size of the image.
    """
    top_right_x = np.max([corners_image[1][0] for corners_image in corners_images])
    bottom_right_x = np.max([corners_images[3][0] for corners_images in corners_images])

    bottom_left_y = np.max([corners_images[2][1] for corners_images in corners_images])
    bottom_right_y = np.max([corners_images[3][1] for corners_images in corners_images])

    width = int(np.ceil(max(bottom_right_x, top_right_x)))
    height = int(np.ceil(max(bottom_right_y, bottom_left_y)))

    width = min(width, 5000)
    height = min(height, 4000)

    return width, height


def get_new_parameters(
    panorama: np.ndarray, image: np.ndarray, H: np.ndarray
) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    Get the new size of the image and the offset matrix.

    Parameters
    ----------
    panorama : np.ndarray
        Current panorama.
    image : np.ndarray
        Image to add to the panorama.
    H : np.ndarray
        Homography matrix for the image.

    Returns
    -------
    size, offset :  Tuple[Tuple[int, int], np.ndarray]
        Size of the new image and offset matrix.
    """
    corners = get_new_corners(image, H)
    added_offset = get_offset(corners)

    corners_image = get_new_corners(image, added_offset @ H)
    if panorama is None:
        size = get_new_size([corners_image])
    else:
        corners_panorama = get_new_corners(panorama, added_offset)
        size = get_new_size([corners_image, corners_panorama])

    return size, added_offset


def single_weights_array(size: int) -> np.ndarray:
    """
    Create a 1D weights array.

    Parameters
    ----------
    size : int
        Size of the array.

    Returns
    -------
    weights : np.ndarray
        1D weights array.
    """
    if size % 2 == 1:
        return np.concatenate(
            [np.linspace(0, 1, (size + 1) // 2), np.linspace(1, 0, (size + 1) // 2)[1:]]
        )
    else:
        return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])


def single_weights_matrix(shape: Tuple[int]) -> np.ndarray:
    """
    Create a 2D weights matrix.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the matrix.

    Returns
    -------
    weights : np.ndarray
        2D weights matrix.
    """
    return (
        single_weights_array(shape[0])[:, np.newaxis]
        @ single_weights_array(shape[1])[:, np.newaxis].T
    )
