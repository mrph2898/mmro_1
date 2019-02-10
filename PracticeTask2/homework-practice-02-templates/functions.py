def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    m = len(x)
    product = None
    if m == 0:
        return 0
    for i in range(min(m, x.size // m)):
        if x[i, i] != 0:
            if product is None:
                product = 1
            product *= x[i, i]
    return product


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    x_l = list(x)
    y_l = list(y)
    if len(x_l) != len(y_l):
        return False
    x_c = x_l[:]
    y_c = y_l[:]
    x_c.sort()
    y_c.sort()
    return (x_c == y_c)


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    x_l = list(x)
    if (x_l.count(0) == 0) or (len(x_l) == 1):
        return None
    return max([x_l[i] for i in range(1, len(x_l)) if x_l[i-1] == 0])


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x num_channels)
    coefs -- 1-d numpy array (length num_channels)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """
    img1 = []
    for i in range(0, len(img)):
        i_list = []
        for j in range(0, len(img[i])):
            coef_sum = 0
            for k in range(0, len(coefs)):
                coef_sum += img[i][j][k] * coefs[k]
            i_list.append(coef_sum)
        img1.append(i_list)
    return img1


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """
    if len(x) == 0:
        return ([], [])
    x_without_repeating = [x[i] for i in range(0, len(x) - 1)
                           if x[1:][i] != x[:-1][i]] + [x[len(x) - 1]]
    array_of_letters_positions_without_repeating =\
        [-1] + [i for i in range(0, len(x) - 1) if x[1:][i] != x[:-1][i]] +\
        [len(x) - 1]
    return (x_without_repeating,
            [array_of_letters_positions_without_repeating[j] -
             array_of_letters_positions_without_repeating[j-1]
             for j in range(1,
                            len(array_of_letters_positions_without_repeating))]
            )


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """
    answer = []
    for i in range(0, len(x)):
        i_list_of_answer = []
        for j in range(0, len(y)):
            distance_ij = 0
            for k in range(0, len(y[j])):
                distance_ij += (x[i][k] - y[j][k]) ** 2
            i_list_of_answer.append(distance_ij ** 0.5)
        answer.append(i_list_of_answer)
    return answer
