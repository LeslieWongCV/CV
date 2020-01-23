from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

# Load image into variable and display it
lion = misc.imread('/Users/leslie/Desktop/革命成果-学术/LENA_FULL.jpg') # Paste address of image
plt.imshow(lion, cmap = plt.get_cmap('gray'))
plt.show()

# Convert color image to grayscale to help extraction of edges and plot it
lion_gray = np.dot(lion[...,:3], [0.299, 0.587, 0.114])
#lion_gray = lion_gray.astype('int32')
plt.imshow(lion_gray, cmap = plt.get_cmap('gray'))
plt.show()