import numpy as np
import cv2
import scipy
from scipy import signal
import math
sigma1 = sigma2 = 1
sum = 0
gaussian = np.zeros([5, 5])
for i in range(5):
    for j in range(5):
        gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - 3) / np.square(sigma1)  # 生成二维高斯分布矩阵
                                            + (np.square(j - 3) / np.square(sigma2)))) / (2 * math.pi * sigma1 * sigma2)
        sum = sum + gaussian[i, j]

Gaussian_op = gaussian / sum

LENA = cv2.imread('/Users/leslie/Desktop/革命成果-学术/LENA_FULL.jpg', 0)

LENA_f = np.copy(LENA)
LENA_f_ = LENA_f.astype('float')
#####################################Step1 高斯模糊#####################################
#Gaussian_op = 1/139*np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]]) #5*5 高斯内核
blur = scipy.signal.convolve(LENA_f_,Gaussian_op,'same')
#####################################Step2 计算梯度幅值和方向#####################################
G_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #Sobel Operator
G_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

X = scipy.signal.convolve2d(blur,G_x,'same')
Y = scipy.signal.convolve2d(blur,G_y,'same')

# X = np.convolve(G_x,LENA_f_,'same')  numpy.convolve 只支持1D卷积
# Y = np.convolve(G_y,LENA_f_,'same')
X_abs = abs(X)
Y_abs = abs(Y)
G = X_abs + Y_abs

sharp = G + LENA_f_
sharp = np.where(sharp<0,0,np.where(sharp>255,255,sharp))
sharp = sharp.astype('uint8')

# Gaussian_op = 1/139*np.array([[12,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]]) #5*5 高斯内核
# blur = scipy.signal.convolve(LENA_f_,Gaussian_op,'same')
# #####################################Step2 计算梯度幅值和方向#####################################
# G_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #Sobel Operator
# G_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
#
# X = scipy.signal.convolve2d(blur,G_x,'same')
# Y = scipy.signal.convolve2d(blur,G_y,'same')
#
# # X = np.convolve(G_x,LENA_f_,'same')  numpy.convolve 只支持1D卷积
# # Y = np.convolve(G_y,LENA_f_,'same')
# X_abs = abs(X)
# Y_abs = abs(Y)
# G = X_abs + Y_abs
#
# sharp1 = G + LENA_f_
# sharp1 = np.where(sharp<0,0,np.where(sharp>255,255,sharp))
# sharp1 = sharp.astype('uint8')

cv2.imshow('Sharp',sharp)
#cv2.imshow('Sharp1',sharp1)

cv2.waitKey()








#blur_ = blur.astype('uint8') #unsigned int 用于显示
#cv2.imshow('blur',blur_)
#cv2.waitKey()
