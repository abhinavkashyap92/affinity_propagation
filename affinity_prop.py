import numpy as np

class AffinityProp(object):
    """

    """
    def __init__(self, similarity_matrix):
        """
            similarity_matrix: N * N matrix containing similarities
        """
        self.s = similarity_matrix

        #  INIITALISE THE RESPONSIBILITY AND THE AVAILABILITY MATRICES
        N, N = self.s.shape
        self.r = np.zeros((N, N))
        self.a = np.zeros((N, N))

    def step(self):
        """
            This is meant ot return the new Availability and repsonsibility matrices
        """
        pass
