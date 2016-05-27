import numpy as np

class AffinityProp(object):
    """
        Implementing the affinity propagation algorithm
    """
    def __init__(self, similarity_matrix, max_iteration=200, num_iter=5,
    alpha=0.5, verbose=True, print_every=100):
        """
            similarity_matrix: N * N matrix containing similarities
            max_iteration: The maximum number of iterations to perfrom for clustering
            num_iter: num iterations without no change in the number of clusters
            that stops the algorithm
        """
        self.s = similarity_matrix
        self.max_iteration = max_iteration
        self.alpha = alpha
        self.verbose = verbose
        self.print_every = print_every

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
        old_r = self.r
        old_a = self.a

        #R UPDATE STEP
        a_plus_s = self.a + self.s
        first_max = np.max(a_plus_s, axis=1)
        first_max_indices = np.argmax(a_plus_s, axis=1)
        first_max = np.repeat(first_max, N).reshape(N, N)
        a_plus_s[range(N), first_max_indices] = -np.inf
        second_max =  np.max(a_plus_s, axis=1)
        r = self.s - first_max
        r[range(N), first_max_indices] = self.s[range(N), first_max_indices] - second_max[range(N)]
        r = self.alpha * old_r + (1 - self.alpha) * r

        # A UPDATE STEP
        rp = np.maximum(r, 0)
        np.fill_diagonal(rp, np.diag(r))
        a = np.repeat(np.sum(rp, axis=0), N).reshape(N,N).T - rp
        da = np.diag(a)
        a = np.minimum(a, 0)
        np.fill_diagonal(a, da)
        a = self.alpha * old_a + (1 - self.alpha) * a

        return r, a

    def solve(self):
        """
            This runs the affinity propagation algorithm until convergence
            TODO: make it into a separate file with more options for better update
            rules and more options
        """
        for i in xrange(self.max_iteration):

            if self.verbose and i % self.print_every is 0:
                print "processing iteration %d" % (i, )
            self.r, self.a = self._step()

        e = self.r + self.a
        N, N = e.shape

        # NOTE: THIS IS ACCORDING TO THE PAPER
        # THIS IS NOT REMOTELY RELATED TO SCIKIT LEARN
        # SO I REALLY CANT COMPARE MY RESULTS TO SCIKIT LEARN BEYOND THIS POINT



        # I will contain the index of the data point that will be an exemplar
        # For example 40 and 55 of say 60 points may serve as the exemplars
        I = np.where(np.diag(e) > 0)[0]
        K = len(I)

        # Select all the rows of S where column_index = 40 and 55
        c = self.s[:, I]
        # For every data point chose the exemplar that has maximum similarity with it
        # For example 1st data point may have maximum similarity with only 44
        # 2nd data point may have max similarity with 55
        # One explanation of why this may be done is to ensure that kth data
        # point not only maximises the sum a+r but also the similarity that it
        # has with i
        # so every c will be either 0 or 1 (considering only 2 exemplars are there)
        c = np.argmax(c, axis=1)

        # Make the c[exemplar_1] = 0 and c[examplar_1] = 1
        c[I] = np.arange(0, K)

        # Get back the index to the original data set
        # say c= [0, 1, 1, 0]
        # 0th exemplar point is 40 in the original data and 1 is 55
        # mapping from c -> I is done like this in numpy
        idx = I[c]

        exemplar_indices = I
        exemplar_assignments =  idx
        return exemplar_indices, exemplar_assignments


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
