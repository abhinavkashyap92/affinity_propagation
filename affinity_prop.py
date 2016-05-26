import numpy as np

class AffinityProp(object):
    """
        Implementing the affinity propagation algorithm
    """
    def __init__(self, similarity_matrix, max_iteration=200, num_iter=5,
    alpha=0.5):
        """
            similarity_matrix: N * N matrix containing similarities
            max_iteration: The maximum number of iterations to perfrom for clustering
            num_iter: num iterations without no change in the number of clusters
            that stops the algorithm
        """
        self.s = similarity_matrix
        self.max_iteration = max_iteration
        self.alpha = alpha

        #  INIITALISE THE RESPONSIBILITY AND THE AVAILABILITY MATRICES
        N, N = self.s.shape
        self.r = np.zeros((N, N))
        self.a = np.zeros((N, N))

    def _step(self):
        """
            This is meant ot return the new Availability and repsonsibility
            matrices for all the data points
        """
        N, N = self.s.shape

        #R UPDATE STEP
        a_plus_s = self.a + self.s
        first_max = np.max(a_plus_s, axis=1)
        first_max_indices = np.argmax(a_plus_s, axis=1)
        first_max = np.repeat(first_max, N).reshape(N, N)
        a_plus_s[range(N), first_max_indices] = -np.inf
        second_max =  np.max(a_plus_s, axis=1)
        r = self.s - first_max
        r[range(N), first_max_indices] = self.s[range(N), first_max_indices] - second_max[range(N)]

        # A UPDATE STEP
        rp = np.maximum(r, 0)
        np.fill_diagonal(rp, np.diag(r))
        a = np.repeat(np.sum(rp, axis=0), N).reshape(N,N).T - rp
        da = np.diag(a)
        a = np.minimum(a, 0)
        np.fill_diagonal(a, da)

        print a
        return r, a

    def solve(self):
        """
            This runs the affinity propagation algorithm until convergence
            TODO: make it into a separate file with more options for better update
            rules and more options
        """
        for i in xrange(self.max_iteration):
            old_r = self.r
            old_a = self.a
            self.r, self.a = self._step()

            # This does something like the adagrad upgrade
            self.r = self.alpha * self.r + (1 - self.alpha) * old_r
            self.a = self.alpha * self.a + (1 - self.alpha) * old_a


if __name__ == "__main__":
    similarity_matrix = np.arange(1, 10).reshape(3, 3)
    max_iteration = 1
    affinity_prop = AffinityProp(similarity_matrix, max_iteration=max_iteration,
                    alpha = 1)
    affinity_prop.solve()

    # checking whether after the first iteration the update to r is proper
    # The correct responsibility matrix was got from working by hand

    correct_responsibility = np.array([[-2, -1, 1], [-2, -1, 1], [-2, -1, 1]], dtype='float')
    is_responsibility_correct = np.allclose(correct_responsibility, affinity_prop.r)
    print "Is the responsiblity correct %s" % (is_responsibility_correct, )

    correct_availability =  np.array([[0, -1, 0], [-2, 0, 0], [-2, -1, 2]])
    is_availability_correct = np.allclose(correct_availability, affinity_prop.a)
    print "Is the availability correct %s" % (is_availability_correct)
