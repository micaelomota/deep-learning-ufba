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

data, labels, classes = dataloader.loadData(TRAIN_PATH)

resized = dataloader.resize(data, IMAGE_WIDTH, IMAGE_HEIGHT)

td, tl, vd, vl = dataloader.splitValidation(resized, labels, 10)

td = np.reshape(td, (len(td), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))/255 
vd = np.reshape(vd, (len(vd), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))/255 

# cv2.imshow('resized', resized[10])
# cv2.waitKey(0)
# exit()

# X_train, y_train, classes_train = load_multiclass_dataset(TRAIN_FOLDER, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
# X_train = X_train/255.#.reshape(-1, IMAGE_HEIGHT*IMAGE_WIDTH*NUM_CHANNELS)/255.
# X_train, y_train = shuffle(X_train, y_train, seed=42)
#X_train, y_train, X_val, y_val = split(X_train, y_train, SPLIT_RATE)
#print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

graph = tf.Graph()
with graph.as_default():
	X = tf.placeholder(tf.float32, shape = (None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
	learning_rate = tf.placeholder(tf.float32)
	is_training = tf.placeholder(tf.bool)
	print(X.shape)

	with tf.variable_scope('encoder'):
		out = tf.layers.conv2d(X, 4, (3, 3), (1, 1), padding='same', activation=tf.nn.relu)
		print(out.shape)
		out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), padding='same')
		print(out.shape)
		out = tf.layers.conv2d(out, 16, (3, 3), (1, 1), padding='same', activation=tf.nn.relu)
		print(out.shape)
		out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), padding='same')
		print(out.shape)
	with tf.variable_scope('decoder'):
		out = tf.layers.conv2d_transpose(out, 4, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
		print(out.shape)
		out = tf.layers.conv2d_transpose(out, 1, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
		print(out.shape)

	decoder_variables = [v for v in tf.global_variables() if v.name.startswith('decoder')]
	encoder_variables = [v for v in tf.global_variables() if v.name.startswith('encoder')]

	#print(encoder_variables, '\n\n\n\n')
	#print(decoder_variables)

	loss = tf.reduce_mean(tf.reduce_sum((out-X)**2))

	encoder_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=encoder_variables)
	decoder_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=decoder_variables)

def training_epoch(session, lr):
	batch_list = np.random.permutation(len(td))

	start = time.time()
	train_loss1 = 0
	train_loss2 = 0
	for j in range(0, len(td), BATCH_SIZE):
		if j+BATCH_SIZE > len(td):
			break
		X_batch = td.take(batch_list[j:j+BATCH_SIZE], axis=0)

		ret1 = session.run([encoder_train_op, loss], feed_dict = {X: X_batch, learning_rate: lr, is_training: True})
		ret2 = session.run([decoder_train_op, loss], feed_dict = {X: X_batch, learning_rate: lr, is_training: True})
		train_loss1 += ret1[1]*BATCH_SIZE
		train_loss2 += ret2[1]*BATCH_SIZE

	pass_size = (len(td)-len(td)%BATCH_SIZE)
	print('Training Epoch:'+str(epoch)+' LR:'+str(lr)+' Time:'+str(time.time()-start)+' Loss1:'+str(train_loss1/pass_size)+' Loss2:'+str(train_loss2/pass_size))

def evaluation(session, Xv, yv, name='Evaluation'):
	start = time.time()
	eval_loss = 0
	eval_acc = 0
	for j in range(0, len(Xv), BATCH_SIZE):
		ret = session.run([loss, correct], feed_dict = {X: Xv[j:j+BATCH_SIZE], y: yv[j:j+BATCH_SIZE], is_training: False})
		eval_loss += ret[0]*min(BATCH_SIZE, len(Xv)-j)
		eval_acc += ret[1]

	print(name+' Epoch:'+str(epoch)+' Time:'+str(time.time()-start)+' ACC:'+str(eval_acc/len(Xv))+' Loss:'+str(eval_loss/len(Xv)))
	return eval_acc/len(Xv), eval_loss/len(Xv)

NUM_EPOCHS_FULL = 200
S_LEARNING_RATE_FULL = 0.001
F_LEARNING_RATE_FULL = 0.001
BATCH_SIZE = 64

with tf.Session(graph = graph) as session:
	# weight initialization
	session.run(tf.global_variables_initializer())

	for epoch in range(NUM_EPOCHS_FULL):
		lr = (S_LEARNING_RATE_FULL*(NUM_EPOCHS_FULL-epoch-1)+F_LEARNING_RATE_FULL*epoch)/(NUM_EPOCHS_FULL-1)
		training_epoch(session, lr)

		# val_acc, val_loss = evaluation(session, X_val, y_val, name='Validation')
		# cv2.imshow('input', td[0].reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
		# rec = session.run(out, feed_dict = {X: td[0:0+1], is_training: False})
		# cv2.imshow('output', rec[0].reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
		# cv2.waitKey(0)
