import numpy as np

def get_euclidean_similarity(data):
    """
        Very naive implementation of getting the similarity matrix of data
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
    similarity = np.zeros((N, N))
    for i in xrange(N):
        for j in xrange(N):
            if i is not j:
                similarity[i][j] = np.linalg.norm(data[i]-data[j])

    # Using the masked array to find the median of the similarities
    masked_array = np.zeros((N, N), dtype='Bool')
    np.fill_diagonal(masked_array, True) # make the diagonals invalid
    ma_similarity = np.ma.masked_array(similarity, masked_array)
    np.fill_diagonal(similarity, np.ma.median(ma_similarity, axis=1))

    return similarity


if __name__ == "__main__":
    # Testing the euclidean similarity with simple matrices
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype='float')
    correct_output = np.array([[8.0, 8.0], [8.0, 8.0]])
    euclidean_similarity = get_euclidean_similarity(data)
    print "Are the two arrays equal %s" % (np.allclose(euclidean_similarity, correct_output),)
