import numpy as np

A = [[1,2],[3,4]]
G1 = [[1,0],[0,-1]]
G2 = [[0,1],[-1,0]]
A = np.array(A)
G1 = np.array(G1)

B = np.convolve(A,A,'valid')
