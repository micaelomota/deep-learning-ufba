import tensorflow as tf
import numpy as np
import random
import time
import sys
import os
import cv2

from src import dataloader

TRAIN_PATH = "data_part1/train/"

IMAGE_HEIGHT = 64  # height of the image
IMAGE_WIDTH = 64   # width of the image
NUM_CHANNELS = 1   # number of channels of the image

NUM_EPOCHS_FULL = 200
S_LEARNING_RATE_FULL = 0.001
F_LEARNING_RATE_FULL = 0.001
BATCH_SIZE = 64

data, labels, classes = dataloader.loadData(TRAIN_PATH)

resized = dataloader.resize(data, IMAGE_WIDTH, IMAGE_HEIGHT)
# cv2.imshow('resized', resized[10])
# cv2.waitKey(0)
# exit()
td, tl, vd, vl = dataloader.splitValidation(resized, labels, 10)

td = np.reshape(td, (len(td), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))/255 
vd = np.reshape(vd, (len(vd), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))/255 

discClasses = np.array(['fake', 'true'])

graph = tf.Graph()
with graph.as_default():
	
	# image input
	X = tf.placeholder(tf.float32, shape = (None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
	
	# discrimator y 
	y = tf.placeholder(tf.int64, shape = (None,))
	y_one_hot = tf.one_hot(y, len(discClasses))
	#print(y_one_hot.shape)

	# generator input
	noise = tf.placeholder(tf.float32, shape = (None, 8, 8, NUM_CHANNELS))
	
	learning_rate = tf.placeholder(tf.float32)

	is_training = tf.placeholder(tf.bool)
	# print(X.shape)

	with tf.variable_scope('discriminator'):
		print("discriminator")
		# conv layer 1
		convL1 = tf.layers.conv2d(X, 32, (3, 3), (1, 1), padding='valid', activation=tf.nn.relu)
		# pooling layer 1
		pool1 = tf.layers.max_pooling2d(inputs=convL1, pool_size=[2, 2], strides=2)
		# conv layer 2
		convL2 = tf.layers.conv2d(pool1, 64, (5, 5), (1, 1), padding='valid', activation=tf.nn.relu)
		# pooling layer 2
		pool2 = tf.layers.max_pooling2d(inputs=convL2, pool_size=[2, 2], strides=2)
		# reshape
		pool2_flat = tf.reshape(pool2, [-1, 13 * 13 * 64])
		#print(pool2_flat.shape)
		# fully conected layer
		fc = tf.layers.dense(pool2_flat, 128, activation=tf.nn.relu)
		# output layer
		out = tf.layers.dense(fc, len(discClasses), activation=tf.nn.sigmoid)
		print(out.shape)

	with tf.variable_scope('generator'):

		print("generator")
		out = tf.layers.conv2d_transpose(noise, 4, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
		out = tf.layers.conv2d_transpose(out, 1, (3, 3), (4, 4), padding='same', activation=tf.nn.relu)
		print(out.shape)

	
	disc_variables = [v for v in tf.global_variables() if v.name.startswith('discriminator')]
	gen_variables = [v for v in tf.global_variables() if v.name.startswith('generator')]


	discloss = tf.reduce_mean(tf.reduce_sum((y_one_hot-out)**2))
	genloss = tf.reduce_mean(tf.reduce_sum((out-X)**2))
	
	discriminator_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(discloss, var_list=disc_variables)
	generator_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(genloss, var_list=gen_variables)



def training_epoch(session, lr):
	batch_list = np.random.permutation(len(td))

	start = time.time()
	train_loss1 = 0
	train_loss2 = 0
	for j in range(0, len(td), BATCH_SIZE):
		if j+BATCH_SIZE > len(td):
			break
		
		X_batch = td.take(batch_list[j:j+BATCH_SIZE], axis=0)

		# Generate noise to feed to the generator
		noise_temp = np.random.uniform(-1., 1., size=[BATCH_SIZE, 8, 8, 1])
		
		# Train generator
		#_, gl, genOut = session.run([generator_op, genloss, out], feed_dict = {X: X_batch, noise: noise_temp, learning_rate: lr, is_training: True})
		
		discTrueY = np.zeros((len(X_batch)))
		#discFakeY = np.zeros((len(X_batch), 2))

		for i in range(len(X_batch)):
			discTrueY[i] = 1
		#	discFakeY[i][0] = 1
		#print(discTrueY, discFakeY)

		# discInputTrain = np.append(X_batch, genOut, axis=0)
		# discInputY = np.zeros((len(discInputTrain), 2) , dtype=np.int) # 0: fake, 1: true
		# print(discInputTrain.shape)
		# print(discInputY.shape)

		ret1 = session.run([discriminator_op, discloss, y_one_hot], feed_dict = {X: X_batch, y: discTrueY, learning_rate: lr, is_training: True})
		print(ret1)
		# train_loss1 += ret1[1]*BATCH_SIZE
		# train_loss2 += ret2[1]*BATCH_SIZE

	# pass_size = (len(td)-len(td)%BATCH_SIZE)
	# print('Training Epoch: '+str(epoch)+' LR: '+str(lr)+' Time: '+str(time.time()-start)+' Loss1: '+str(train_loss1/pass_size)+' Loss2: '+str(train_loss2/pass_size))



with tf.Session(graph = graph) as session:
	# weight initialization
	session.run(tf.global_variables_initializer())

	for epoch in range(NUM_EPOCHS_FULL):
		lr = (S_LEARNING_RATE_FULL*(NUM_EPOCHS_FULL-epoch-1)+F_LEARNING_RATE_FULL*epoch)/(NUM_EPOCHS_FULL-1)
		training_epoch(session, lr)
		exit()

	# 	# if (epoch+1)%10 == 0:
	# 	print("saving images...")
	# 	batch_list = np.random.permutation(len(td))
	# 	X_batch = td.take(batch_list[10:10+int(BATCH_SIZE/2)], axis=0)

	# 	rec = session.run(out, feed_dict = {X: X_batch, is_training: False})
	# 	print(rec.shape)
	# 	cv2.imshow('output', rec[0].reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
	# 	cv2.waitKey(0)

		
	# 	exit()
		# for i in range(len(rec)):
		# 	cv2.imwrite('generated/img'+str(i)+'.png', rec[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH))

		# val_acc, val_loss = evaluation(session, X_val, y_val, name='Validation')
		# cv2.imshow('input', td[0].reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
		# rec = session.run(out, feed_dict = {X: td[0:0+1], is_training: False})
		# cv2.imshow('output', rec[0].reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
		# cv2.waitKey(0)