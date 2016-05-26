import numpy as np
from affinity_prop import *
from utils import *

if __name__ == "__main__":
    # DATA DESCRIPTION
    # Iris are a kind of plants
    # There are three species in the dataset given by the last column
    data = np.genfromtxt('iris.data', delimiter=",")
    N, M = data.shape
    similarity_matrix = get_euclidean_similarity(data)

    affinity_propagation =  AffinityProp(similarity_matrix)
    affinity_propagation.solve()
