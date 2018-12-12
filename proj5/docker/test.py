# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import tensorflow as tf
import numpy as np
import cv2
import sys

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Load the test set                                                                                  #
# ---------------------------------------------------------------------------------------------------------- #
with open(sys.argv[1], 'r') as f:
	l = [line.split() for line in f]
zs = [[float(x) for x in line[:-1]] for line in l]
X_name = [line[-1] for line in l]
X_test = np.array(zs).reshape(-1, 8, 8, 1)
print(X_name)
print(X_test.shape, X_test.dtype)

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Inference graph                                                                                    #
# ---------------------------------------------------------------------------------------------------------- #
graph = tf.Graph()
with graph.as_default():
	X = tf.placeholder(tf.float32, shape = (None, 8, 8, 1))

	out = tf.layers.conv2d_transpose(X, 4, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
	print(out.shape)
	out = tf.layers.conv2d_transpose(out, 16, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
	print(out.shape)
	result = tf.layers.conv2d_transpose(out, 1, (3, 3), (2, 2), padding='same', activation=tf.nn.sigmoid)
	print(result.shape)

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Test loop                                                                                          #
# ---------------------------------------------------------------------------------------------------------- #
with tf.Session(graph = graph) as session:
	session.run(tf.global_variables_initializer())

	for i in range(len(X_name)):
		ret = session.run([result], feed_dict = {X: X_test[i:i+1]})
		cv2.imwrite(sys.argv[2]+X_name[i], ret[0].reshape(64,64))

