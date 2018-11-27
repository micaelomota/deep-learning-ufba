import tensorflow as tf
import numpy as np
from src import dataloader
import time
import cv2

IMAGE_HEIGHT = 77
IMAGE_WIDTH = 71
NUM_CHANNELS = 1

TRAIN_PATH = "data_part1/train/"

data, labels, classes = dataloader.loadData(TRAIN_PATH)
td, tl, vd, vl = dataloader.splitValidation(data, labels, 50)

# normalizing data
# td = np.reshape(td, (len(td), 77*71))/255
# vd = np.reshape(vd, (len(vd), 77*71))/255
td = np.reshape(td, (len(td), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))/255 
vd = np.reshape(vd, (len(vd), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))/255 

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
	y_one_hot = tf.one_hot(y, len(classes))
	learning_rate = tf.placeholder(tf.float32)
		
	out = tf.layers.dense(dropout, len(classes), activation=tf.nn.sigmoid)

	loss = tf.reduce_mean(tf.reduce_sum((y_one_hot-out)**2))
	train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
	
	result = tf.argmax(out, 1)
	correct = tf.reduce_sum(tf.cast(tf.equal(result, y), tf.float32))
	


def training_epoch(epoch, session, op, lr):
	batch_list = np.random.permutation(len(td))
	start = time.time()
	train_loss = 0
	train_acc = 0
	for j in range(0, len(td), BATCH_SIZE):
		
		if j+BATCH_SIZE > len(td):
			break
		X_batch = td.take(batch_list[j:j+BATCH_SIZE], axis=0)
		y_batch = tl.take(batch_list[j:j+BATCH_SIZE], axis=0)

		# run augmentation here
		rotated = dataloader.rotate_images(X_batch)
		
		for k in range(len(rotated)):
			cv2.imshow('original ' + str(y_batch[k]), X_batch[k])
			cv2.imshow('rotated ' + str(y_batch[k]), rotated[k])
		
		cv2.waitKey(0)
		cv2.destroyAllWindows()


		ret = session.run([op, loss, correct], feed_dict = {
			X: X_batch, 
			y: y_batch, 
			learning_rate: lr,
			is_train: 1
		})
		
		train_loss += ret[1]*BATCH_SIZE
		train_acc += ret[2]

	pass_size = (len(td)-len(td)%BATCH_SIZE)
	print('Training Epoch: '+str(epoch)+' LR: '+str(lr)+' Time: '+str(time.time()-start)+' ACC: '+str(train_acc/pass_size)+' Loss: '+str(train_loss/pass_size))


def evaluation(epoch, session, Xv, yv, name='Evaluation'):
	start = time.time()
	eval_loss = 0
	eval_acc = 0
	for j in range(0, len(Xv), BATCH_SIZE):
		ret = session.run([loss, correct], feed_dict = {X: Xv[j:j+BATCH_SIZE], y: yv[j:j+BATCH_SIZE], is_train: 0})
		eval_loss += ret[0]*min(BATCH_SIZE, len(Xv)-j)
		eval_acc += ret[1]

	print(name+' Epoch: '+str(epoch)+' Time: '+str(time.time()-start)+' ACC: '+str(eval_acc/len(Xv))+' Loss: '+str(eval_loss/len(Xv)))

	return eval_acc/len(Xv), eval_loss/len(Xv)

NUM_EPOCHS_FULL = 50
S_LEARNING_RATE_FULL = 0.01
F_LEARNING_RATE_FULL = 0.0001
BATCH_SIZE = 16

writerLoss = tf.summary.FileWriter("./logs/project4/loss_")
writerAcc = tf.summary.FileWriter("./logs/project4/acc_")
log_var = tf.Variable(0.0)
tf.summary.scalar("train", log_var)

write_op = tf.summary.merge_all()
plotSession = tf.InteractiveSession()
plotSession.run(tf.global_variables_initializer())

def train():
	with tf.Session(graph = graph) as session:
		# weight initialization
		session.run(tf.global_variables_initializer())

		# full optimization
		maxAcc = 0
		for epoch in range(NUM_EPOCHS_FULL):
			lr = (S_LEARNING_RATE_FULL*(NUM_EPOCHS_FULL-epoch-1)+F_LEARNING_RATE_FULL*epoch)/(NUM_EPOCHS_FULL-1)
			training_epoch(epoch, session, train_op, lr)

			val_acc, val_loss = evaluation(epoch, session, vd, vl, name='Validation')

			if val_acc > maxAcc:
				maxAcc = val_acc
				save_path = tf.train.Saver().save(session, "models/tf-project4/model.ckpt")
				print("Model with acc: {} saved in path {}".format(maxAcc,save_path))

			summary = plotSession.run(write_op, {log_var: val_acc})
			writerAcc.add_summary(summary, epoch)
			writerAcc.flush()

			summary = plotSession.run(write_op, {log_var: val_loss})
			writerLoss.add_summary(summary, epoch)
			writerLoss.flush()

		

def runInference():
	data, names = dataloader.loadTestData('data_part1/test/')
	#rdata = np.reshape(data, (len(data), 77*71))/255
	rdata = np.reshape(data, (len(data), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))/255 
	output_path = "tfproject4.txt"

	output = open(output_path, "w") 
	with tf.Session(graph = graph) as session:
		s = tf.train.Saver().restore(session, "models/tf-project4/model.ckpt")
		print("model loaded")
		for i in range(len(data)):
			
			ret = session.run([result], feed_dict = { X: np.array([rdata[i]]), is_train: 0})
			output.write("{} {}\n".format(names[i], ret[0][0]))
			# cv2.imshow(names[i], data[i])
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			#exit()
	output.close()
	print("output saved file: " + output_path)

print("======= PROJECT 4 ==========")

todo = "_"
while todo != "train" and todo != "inference":
	todo = input("What do you want to do? (1:train, 2:inference other: quit): ")
	if todo == "1":
		print("Trainning...")
		train()
	elif todo == "2":
		print("Running inference...")
		runInference()
	else:
		exit()