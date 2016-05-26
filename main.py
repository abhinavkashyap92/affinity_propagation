import numpy as np
from affinity_prop import *
from utils import *

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import euclidean_distances
from sklearn.cluster.affinity_propagation_ import affinity_propagation

if __name__ == "__main__":

    n_clusters = 3
    centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10
    X, _ = make_blobs(n_samples=60, n_features=2, centers=centers,
                  cluster_std=0.4, shuffle=True, random_state=0)
    S = -1 * euclidean_distance(X, squared=True)
    median_value = np.median(S) * 10
    np.fill_diagonal(S, median_value)

    af_prop = AffinityProp(S)
    exemplar_indices, exemplar_assignments = af_prop.solve()
    print "cluster center indices mine: %s" % (exemplar_indices,)
    print len(exemplar_assignments)
    print np.unique(exemplar_assignments, return_counts=True)

    # RESULTS
    # THE INDICES OF THE EXEMPLARS IS 4 21 AND 22
    # THERE ARE 19, 20, 21 ELEMENTS IN EACH CLUSTER
