import numpy as np

def euclidean_distance(data, squared=True):
    """
        ARGS:
            INPUT:
                data: A matrix of shape (N, M)
                N -> Number of data points
                M -> Number of dimensions
            OUTPUT:
                similarity: of shape N * N representing the euclidean
                distance between the ith sample with all other samples
                Also here the sel similarity of the data point i is set to
                the median of the similarities with other data points
    """
    N, M = data.shape
    data = data.astype('float') # just making sure that it is float
    data_copy = data.copy()
    distance = np.zeros((N, N))

    data_square = np.sum(np.square(data), axis=1)
    data_copy_square = np.sum(np.square(data_copy), axis=1)
    multiply = np.dot(data, data_copy.T)
    distance = data_square[:, np.newaxis] + data_copy_square - (2 * multiply)

    if not squared:
        distance = np.sqrt(distance)

    return distance


if __name__ == "__main__":
    # Testing the euclidean similarity with simple matrices

    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype='float')
    N, M = data.shape
    correct_output = np.array([[0.0, 8.0], [8.0, 0.0]])
    similarity = euclidean_distance(data, square=False)
    print "Are the two similarities equal %s" % (np.allclose(similarity, correct_output),)

    # Filling the diagonal of the euclidean distance
    # to the median but not including the self similarity points themselves
    # Using the masked array to find the median of the similarities
    masked_array = np.zeros((N, N), dtype='Bool')
    np.fill_diagonal(masked_array, True) # make the diagonals invalid
    ma_similarity = np.ma.masked_array(similarity, masked_array)
    np.fill_diagonal(similarity, np.ma.median(ma_similarity))
    correct_similarity_with_median = np.array([[8.0, 8.0], [8.0, 8.0]])
    print "Is similarities with medians equal %s" % (np.allclose(similarity, correct_similarity_with_median))
