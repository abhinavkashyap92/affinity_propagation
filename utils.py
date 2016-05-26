import numpy as np

def euclidean_distance(data, square=True):
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
    similarity = np.zeros((N, N))

    data_square = np.sum(np.square(data), axis=1)
    data_copy_square = np.sum(np.square(data_copy), axis=1)
    multiply = np.dot(data, data_copy.T)
    similarity = data_square[:, np.newaxis] + data_copy_square - (2 * multiply)

    if not square:
        similarity = np.sqrt(similarity)

    return similarity


if __name__ == "__main__":
    # Testing the euclidean similarity with simple matrices
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype='float')
    correct_output = np.array([[0.0, 8.0], [8.0, 0.0]])
    euclidean_similarity = euclidean_distance(data, square=False)
    print euclidean_similarity
    print "Are the two arrays equal %s" % (np.allclose(euclidean_similarity, correct_output),)
