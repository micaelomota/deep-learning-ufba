import tensorflow as tf
import numpy as np
from src import dataloader
import time

data, labels, classes = dataloader.loadData("data_part1/train/")
td, tl, vd, vl = dataloader.splitValidation(data, labels, 10)

# normalizing data
td = np.reshape(td, (len(td), 77*71))/255
vd = np.reshape(vd, (len(vd), 77*71))/255

# # 10 dimensoes para o y
# tl10 = np.zeros((len(tl), 10))
# for i in range(len(tl)):
#     tl10[i][int(tl[i])] = 1

# vl10 = np.zeros((len(vl), 10))
# for i in range(len(vl)):
#     vl10[i][int(vl[i])] = 1

# print(td.shape, tl.shape, vd.shape, vl.shape)
# exit()



graph = tf.Graph()
with graph.as_default():
	X = tf.placeholder(tf.float32, shape = (None, len(td[0])))
	y = tf.placeholder(tf.int64, shape = (None,))
	
	y_one_hot = tf.one_hot(y, len(classes))
	learning_rate = tf.placeholder(tf.float32)

	#fc = tf.layers.dense(X, 512, activation=tf.nn.relu)
	out = tf.layers.dense(X, len(classes), activation=tf.nn.sigmoid)

	loss = tf.reduce_mean(tf.reduce_sum((y_one_hot-out)**2))
	train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
	
	result = tf.argmax(out, 1)
	correct = tf.reduce_sum(tf.cast(tf.equal(result, y), tf.float32))
	


def training_epoch(session, op, lr):
	batch_list = np.random.permutation(len(td))
	start = time.time()
	train_loss = 0
	train_acc = 0
	for j in range(0, len(td), BATCH_SIZE):
		
		if j+BATCH_SIZE > len(td):
			break
		X_batch = td.take(batch_list[j:j+BATCH_SIZE], axis=0)
		y_batch = tl.take(batch_list[j:j+BATCH_SIZE], axis=0)

		ret = session.run([op, loss, correct], feed_dict = {X: X_batch, y: y_batch, learning_rate: lr})
		
		train_loss += ret[1]*BATCH_SIZE
		train_acc += ret[2]

	pass_size = (len(td)-len(td)%BATCH_SIZE)
	print('Training Epoch: '+str(epoch)+' LR: '+str(lr)+' Time: '+str(time.time()-start)+' ACC: '+str(train_acc/pass_size)+' Loss: '+str(train_loss/pass_size))


def evaluation(session, Xv, yv, name='Evaluation'):
	start = time.time()
	eval_loss = 0
	eval_acc = 0
	for j in range(0, len(Xv), BATCH_SIZE):
		ret = session.run([loss, correct], feed_dict = {X: Xv[j:j+BATCH_SIZE], y: yv[j:j+BATCH_SIZE]})
		eval_loss += ret[0]*min(BATCH_SIZE, len(Xv)-j)
		eval_acc += ret[1]

	print(name+' Epoch: '+str(epoch)+' Time: '+str(time.time()-start)+' ACC: '+str(eval_acc/len(Xv))+' Loss: '+str(eval_loss/len(Xv)))

	return eval_acc/len(Xv), eval_loss/len(Xv)


NUM_EPOCHS_FULL = 50
S_LEARNING_RATE_FULL = 0.01
F_LEARNING_RATE_FULL = 0.0001
BATCH_SIZE = 32

with tf.Session(graph = graph) as session:
	# weight initialization
	session.run(tf.global_variables_initializer())

	# full optimization
	for epoch in range(NUM_EPOCHS_FULL):
		lr = (S_LEARNING_RATE_FULL*(NUM_EPOCHS_FULL-epoch-1)+F_LEARNING_RATE_FULL*epoch)/(NUM_EPOCHS_FULL-1)
		training_epoch(session, train_op, lr)

		val_acc, val_loss = evaluation(session, vd, vl, name='Validation')

	save_path = tf.train.Saver().save(session, "models/tf-lr/model.ckpt")
	print("Model saved in path: %s" % save_path)