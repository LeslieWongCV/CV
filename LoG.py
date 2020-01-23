import numpy as np
import cv2
import scipy.ndimage as nd
LENA = cv2.imread('/Users/leslie/Desktop/革命成果-学术/LENA_FULL.jpg', 0)


LENA_f = np.copy(LENA)
LENA_f_ = LENA_f.astype('float')

Result = nd.gaussian_laplace(LENA_f_, sigma=1)   #scipy.ndimage 模块中的高斯拉普拉斯算子

cv2.imshow('LENA',Result)
cv2.waitKey()



