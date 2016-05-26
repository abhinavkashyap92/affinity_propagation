import numpy as np

if __name__ == "__main__":
    # DATA DESCRIPTION
    # Iris are a kind of plants
    # There are three species in the dataset given by the last column
    data = np.genfromtxt('iris.data', delimiter=",")
    print data.shape 
