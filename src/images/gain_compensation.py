import numpy as np


def get_gain_compensations(images, pair_matches, sigma_n: float = 10.0, sigma_g: float = 0.1):
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

    gains_normalized = gains / gains.max()

    return gains_normalized
