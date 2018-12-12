# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import numpy as np
import cv2
import os

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Load all images from a multiclass dataset (folder of folders). Each folder inside the main folder  #
#         represents a different class and its name is used as class label. Train and test folders must have #
#         the same directory structure, otherwise labels and their respective indexes will be misaligned.    #
#         All images must have the same size, the same number of channels and 8 bits per channel.            #
# Parameters:                                                                                                #
#         path - path to the main folder                                                                     #
#         height - number of image rows                                                                      #
#         width - number of image columns                                                                    #
#         num_channels - number of image channels                                                            #
# Return values:                                                                                             #
#         X - ndarray with all images                                                                        #
#         y - ndarray with indexes of labels (y[i] is the label for X[i])                                    #
#         l - list of existing labels (1st label in the list has index 0, 1nd has index 1, and so on)        #
# ---------------------------------------------------------------------------------------------------------- #
def load_multiclass_dataset(path, height=224, width=224, num_channels=3):
	classes = sorted(os.listdir(path))
	images = [sorted(os.listdir(path+'/'+id)) for id in classes]
	num_images = np.sum([len(l) for l in images])

	X = np.empty([num_images, height, width, num_channels], dtype=np.uint8)
	y = np.empty([num_images], dtype=np.int64)

	k = 0
	for i in range(len(classes)):
		for j in range(len(images[i])):
			img = cv2.imread(path+'/'+classes[i]+'/'+images[i][j], cv2.IMREAD_GRAYSCALE)
			#assert img.shape == (height, width, num_channels), "%r has an invalid image size!" % images[i][j]
			#assert img.dtype == np.uint8, "%r has an invalid pixel format!" % images[i][j]
			img = cv2.resize(img, (height, width)).reshape(height, width, num_channels)
			X[k] = img
			y[k] = i
			k += 1

	return X, y, classes

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Shuffle the first dimension of two multidimensional arrays simultaneously. The 1st dimension size  #
#         must be the same for both arrays.                                                                  #
# Parameters:                                                                                                #
#         X - data array                                                                                     #
#         y - labels array                                                                                   #
# Return values:                                                                                             #
#         X - shuffled data array                                                                            #
#         y - shuffled labels array                                                                          #
# ---------------------------------------------------------------------------------------------------------- #
def shuffle(X, y, seed=None):
	assert len(X) == len(y), "The 1st dimension size must be the same for both arrays!"
	if seed is not None:
		np.random.seed(seed)
	p = np.random.permutation(len(X))
	return X[p], y[p]

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Split two multidimensional arrays in two parts in the first dimension. The 1st dimension size must #
#         be the same for both arrays.                                                                       #
# Parameters:                                                                                                #
#         X - data array                                                                                     #
#         y - labels array                                                                                   #
#         rate - rate of elements for the 1st part                                                           #
# Return values:                                                                                             #
#         X1 - 1st part of data array                                                                        #
#         y1 - 1st part of labels array                                                                      #
#         X2 - 2nd part of data array                                                                        #
#         y2 - 2nd part of labels array                                                                      #
# ---------------------------------------------------------------------------------------------------------- #
def split(X, y, rate):
	assert len(X) == len(y), "The 1st dimension size must be the same for both arrays!"
	idx = int(len(X)*float(rate))
	return X[:idx], y[:idx], X[idx:], y[idx:]

