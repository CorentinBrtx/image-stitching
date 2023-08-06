import numpy as np

from src.images import Image
from src.matching import PairMatch


def set_gain_compensations(
    images: list[Image], pair_matches: list[PairMatch], sigma_n: float = 10.0, sigma_g: float = 0.1
) -> None:
    """
    Compute the gain compensation for each image, and save it into the images objects.

    Args:
        images: Images of the panorama
        pair_matches: Pair matches between the images
        sigma_n: Standard deviation of the normalized intensity error
        sigma_g: Standard deviation of the gain
    """
    coefficients = []
    results = []

    for k, image in enumerate(images):
        coefs = [np.zeros(3) for _ in range(len(images))]
        result = np.zeros(3)

        for pair_match in pair_matches:
            if pair_match.image_a == image:
                coefs[k] += pair_match.area_overlap * (
                    (2 * pair_match.Iab ** 2 / sigma_n ** 2) + (1 / sigma_g ** 2)
                )

                i = images.index(pair_match.image_b)
                coefs[i] -= (
                    (2 / sigma_n ** 2) * pair_match.area_overlap * pair_match.Iab * pair_match.Iba
                )

                result += pair_match.area_overlap / sigma_g ** 2

            elif pair_match.image_b == image:
                coefs[k] += pair_match.area_overlap * (
                    (2 * pair_match.Iba ** 2 / sigma_n ** 2) + (1 / sigma_g ** 2)
                )

                i = images.index(pair_match.image_a)
                coefs[i] -= (
                    (2 / sigma_n ** 2) * pair_match.area_overlap * pair_match.Iab * pair_match.Iba
                )

                result += pair_match.area_overlap / sigma_g ** 2

        coefficients.append(coefs)
        results.append(result)

    coefficients = np.array(coefficients)
    results = np.array(results)

    gains = np.zeros_like(results)

    for channel in range(coefficients.shape[2]):
        coefs = coefficients[:, :, channel]
        res = results[:, channel]

        gains[:, channel] = np.linalg.solve(coefs, res)

    max_pixel_value = np.max([image.image for image in images])

    if gains.max() * max_pixel_value > 255:
        gains = gains / (gains.max() * max_pixel_value) * 255

    for i, image in enumerate(images):
        image.gain = gains[i]
