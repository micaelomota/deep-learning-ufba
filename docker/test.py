import tensorflow as tf
import numpy as np
from src import dataloader
import time
import cv2
import sys

IMAGE_HEIGHT = 77
IMAGE_WIDTH = 71
NUM_CHANNELS = 1
NUM_EPOCHS_FULL = 50
S_LEARNING_RATE_FULL = 0.01
F_LEARNING_RATE_FULL = 0.0001
BATCH_SIZE = 16


graph = tf.Graph()
with graph.as_default():
	is_train = tf.placeholder(tf.bool, name="is_train")

	# X = tf.placeholder(tf.float32, shape = (None, len(td[0])))
	X = tf.placeholder(tf.float32, shape = (None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))

	# conv layer 1
	convL1 = tf.layers.conv2d(X, 32, (3, 3), (1, 1), padding='valid', activation=tf.nn.relu)

	# pooling layer 1
	pool1 = tf.layers.max_pooling2d(inputs=convL1, pool_size=[2, 2], strides=2)

	# conv layer 2
	convL2 = tf.layers.conv2d(pool1, 64, (5, 5), (1, 1), padding='valid', activation=tf.nn.relu)

	# pooling layer 2
	pool2 = tf.layers.max_pooling2d(inputs=convL2, pool_size=[2, 2], strides=2)

	
	pool2_flat = tf.reshape(pool2, [-1, 16 * 15 * 64])
	# print(pool2_flat.shape)

	# fully conected layer
	fc = tf.layers.dense(pool2_flat, 128, activation=tf.nn.relu)

	dropout = tf.layers.dropout(fc, 0.4, training=is_train)

	y = tf.placeholder(tf.int64, shape = (None,))
	y_one_hot = tf.one_hot(y, 10)
	learning_rate = tf.placeholder(tf.float32)
		
	out = tf.layers.dense(dropout, 10, activation=tf.nn.sigmoid)

	loss = tf.reduce_mean(tf.reduce_sum((y_one_hot-out)**2))
	train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
	
	result = tf.argmax(out, 1)
	correct = tf.reduce_sum(tf.cast(tf.equal(result, y), tf.float32))
		
data, names = dataloader.loadTestData(sys.argv[1])
#rdata = np.reshape(data, (len(data), 77*71))/255
rdata = np.reshape(data, (len(data), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))/255 	
output = open(sys.argv[2], "w") 

with tf.Session(graph = graph) as session:
	s = tf.train.Saver().restore(session, "model/model.ckpt")
	for i in range(len(data)):
		ret = session.run([result], feed_dict = { X: np.array([rdata[i]]), is_train: 0})
		output.write("{} {}\n".format(names[i], ret[0][0]))		
	output.close()

