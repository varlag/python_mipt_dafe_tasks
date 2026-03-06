import numpy as np


def get_extremum_indices(
    ordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if ordinates.size < 3: 
        raise ValueError

    l = ordinates[:-2]
    c = ordinates[1:-1]
    r = ordinates[2:]

    max_e = (l < c) & (c > r)
    min_e = (l > c) & (c < r)
    ans = np.arange(1, ordinates.size - 1)

    return ans[min_e], ans[max_e]
