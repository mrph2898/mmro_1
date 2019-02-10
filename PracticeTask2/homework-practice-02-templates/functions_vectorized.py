import numpy as np


def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """
    diag = x.diagonal()
    mask = (diag != 0)
    if mask.any():
        return np.prod(diag[mask])
    else:
        return None


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """
    if x.shape != y.shape:
        return False
    else:
        unique_x, counts_x = np.unique(x, return_counts=True)
        unique_y, counts_y = np.unique(y, return_counts=True)
        if unique_x.shape[0] != unique_y.shape[0]:
            return False
        return (np.array_equal(unique_x, unique_y) and
                np.array_equal(counts_x, counts_y))


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """
    mask = np.array(np.where(x == 0))
    if mask.size == 0 or x.size == 1:
        return None
    if len(x) - 1 in mask:
        mask = np.delete(mask, -1)
    mask += 1
    return(np.nanmax(x[mask]))


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x num_channels)
    coefs -- 1-d numpy array (length num_channels)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """
    return np.sum(img * coefs, axis=2)


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """
    if len(x) == 0:
        return ([], [])
    y = np.copy(x)
    array_of_letters_positions_without_repeating = \
        np.append(np.where(y[1:] != y[:-1]), len(x) - 1)
    return (y[array_of_letters_positions_without_repeating],
            np.diff(np.append(-1,
                              array_of_letters_positions_without_repeating)))


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """
    return np.sum((x[:, None] - y[None, :]) ** 2, axis=2) ** 0.5


"""
    return np.sqrt((x ** 2).sum(axis=1)[:, None] +
                   (y ** 2).sum(axis=1)[None, :] +
                   (-2) * np.matmul(x, y.T))
"""
