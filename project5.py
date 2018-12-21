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

NUM_EPOCHS_FULL = 2000
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

def generator(noise, reuse=None):
	with tf.variable_scope('generator', reuse=reuse):
		print("generator")
		gen_out = tf.layers.conv2d_transpose(noise, 4, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
		gen_out = tf.layers.conv2d_transpose(gen_out, 1, (3, 3), (4, 4), padding='same', activation=tf.nn.relu)
		print(gen_out.shape)
		return gen_out

def discriminator(X, reuse=None):
	with tf.variable_scope('discriminator', reuse=reuse):
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
		disc_out = tf.layers.dense(fc, 1, activation=tf.nn.sigmoid)
		print(disc_out.shape)
		return disc_out


graph = tf.Graph()
with graph.as_default():
	# image input
	X = tf.placeholder(tf.float32, shape = (None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
	# discriminator y 
	y = tf.placeholder(tf.float32, shape = (None))

	# generator input
	noise = tf.placeholder(tf.float32, shape = (None, 8, 8, NUM_CHANNELS))
	
	learning_rate = tf.placeholder(tf.float32)

	is_training = tf.placeholder(tf.bool)
	# print(X.shape)

	generated = generator(noise)
	realOut = discriminator(X)
	fakeOut = discriminator(generated, reuse=True)

	disc_variables = [v for v in tf.global_variables() if v.name.startswith('discriminator')]
	gen_variables = [v for v in tf.global_variables() if v.name.startswith('generator')]

	genloss = -tf.reduce_mean(tf.log(fakeOut))
	discloss = -tf.reduce_mean(tf.log(realOut)+tf.log(1.-fakeOut))

	generator_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(genloss, var_list=gen_variables)
	discriminator_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(discloss, var_list=disc_variables)

	# disc_real_out=disc_fun(disc_inp)
	# disc_fake_out=disc_fun(gen_out)

	# optim_gen=tf.train.AdamOptimizer(learning_rate=learning_rate)
	# optim_disc=tf.train.AdamOptimizer(learning_rate=learning_rate)
	# cost_gen=-tf.reduce_mean(tf.log(disc_fake_out))
	# cost_disc=-tf.reduce_mean(tf.log(disc_real_out)+tf.log(1.-disc_fake_out))


def training_epoch(session, lr):
	batch_list = np.random.permutation(len(td))

	start = time.time()
	gen_loss = 0
	disc_loss = 0
	genOut = []
	for j in range(0, len(td), BATCH_SIZE):
		if j+BATCH_SIZE > len(td):
			print("saving images")
			for i in range(BATCH_SIZE):
				img = genOut[i]
				cv2.imwrite('generated/img'+str(i)+'.png', img)
			break
		
		X_batch = td.take(batch_list[j:j+BATCH_SIZE], axis=0)

		# Generate noise to feed to the generator
		noise_temp = np.random.uniform(-1., 1., size=[BATCH_SIZE, 8, 8, 1])
		
		_, _, gl, dl, genOut = session.run([generator_op, discriminator_op, genloss, discloss, generated], feed_dict = {X: X_batch, noise: noise_temp, learning_rate: lr, is_training: True})

		print('j: %f, Generator Loss: %f, Discriminator Loss: %f' % (j, gl, dl))

		# print(genOut.shape)

		# Train generator
		# _, gl, genOut = session.run([generator_op, genloss, gen_out], feed_dict = {X: X_batch, noise: noise_temp, learning_rate: lr, is_training: True})
			


		# discTrainBatch = np.append(X_batch, genOut, axis=0) # join batches

		# # Train discriminator
		# _, dl = session.run([discriminator_op, discloss], feed_dict = {X: discTrainBatch, y: discTrueY, learning_rate: lr, is_training: True})
		# # print(dl)

		# gen_loss += gl*BATCH_SIZE
		# disc_loss += dl*(BATCH_SIZE*2)

	#pass_size = (len(td)-len(td)%BATCH_SIZE)
	#print('Epoch: '+str(epoch)+' LR: '+str(lr)+' Time: '+str(time.time()-start)+' Gen loss: '+str(gen_loss/pass_size)+' Disc loss: '+str(disc_loss/(pass_size*2)))



with tf.Session(graph = graph) as session:
	# weight initialization
	session.run(tf.global_variables_initializer())

	for epoch in range(NUM_EPOCHS_FULL):
		lr = (S_LEARNING_RATE_FULL*(NUM_EPOCHS_FULL-epoch-1)+F_LEARNING_RATE_FULL*epoch)/(NUM_EPOCHS_FULL-1)
		training_epoch(session, lr)

		save_path = tf.train.Saver().save(session, "models/model5.ckpt")
		print("Model saved in path {}".format(save_path))

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
