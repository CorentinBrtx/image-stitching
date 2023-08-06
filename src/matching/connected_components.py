from src.matching.pair_match import PairMatch


def find_connected_components(pair_matches: list[PairMatch]) -> list[list[PairMatch]]:
    """
    Find the connected components of the given pair matches.

    Args:
        pair_matches: The list of pair matches.

    Returns:
        connected_components: List of connected components.
    """
    connected_components = []
    pair_matches_to_check = pair_matches.copy()
    component_id = 0
    while len(pair_matches_to_check) > 0:
        pair_match = pair_matches_to_check.pop(0)
        connected_component = {pair_match.image_a, pair_match.image_b}
        size = len(connected_component)
        stable = False
        while not stable:
            i = 0
            while i < len(pair_matches_to_check):
                pair_match = pair_matches_to_check[i]
                if (
                    pair_match.image_a in connected_component
                    or pair_match.image_b in connected_component
                ):
                    connected_component.add(pair_match.image_a)
                    connected_component.add(pair_match.image_b)
                    pair_matches_to_check.pop(i)
                else:
                    i += 1
            stable = size == len(connected_component)
            size = len(connected_component)
        connected_components.append(list(connected_component))
        for image in connected_component:
            image.component_id = component_id
        component_id += 1

    return connected_components
