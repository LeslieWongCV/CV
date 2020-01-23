import matplotlib.pyplot as plt
import numpy as np

A = np.array([[255,255,255,255,255,255],
              [255,1,255,255,255,255],
              [255,1,1,1,255,255],
              [255,255,1,1,255,255],
              [255,255,1,255,255,255],
              [255,255,255,255,255,255]])

B = np.array([[255,255,255,255,255,255],
              [255,255,255,1,255,255],
              [255,255,1,255,255,255],
              [255,1,255,255,255,255],
              [255,255,255,255,255,255],
              [255,255,255,255,255,255]])

Intersection = A + B


def Intersection_F(A):
    for i in range(0,A.shape[0]-1):
        for j in range(1,A.shape[1]):
            if(A[i,j] == 256):
                A[i,j]=1
    return A

Intersection = Intersection_F(Intersection)


plt.imshow(Intersection,cmap=plt.get_cmap('gray'))
plt.show()