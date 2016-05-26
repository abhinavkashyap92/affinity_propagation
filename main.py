import numpy as np
from affinity_prop import *
from utils import *

if __name__ == "__main__":
    # DATA DESCRIPTION
    # Iris are a kind of plants
    # There are three species in the dataset given by the last column
    data = np.genfromtxt('iris.data', delimiter=",")
    N, M = data.shape
    similarity = -1 * euclidean_distance(data)
    # The paper says all the input similarities may be set
    # an equal value which is equal to the median of the similarities
    np.fill_diagonal(similarity, np.median(similarity))
    affinity_propagation =  AffinityProp(similarity_matrix)
    affinity_propagation.solve()
