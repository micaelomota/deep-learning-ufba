import os
import numpy as np
import cv2
import tensorflow as tf
from math import ceil, floor
import uuid 
import random

def loadData(path):
    data = []
    label = []
    classes = []

    folders = sorted(os.listdir(path))
    for folder in folders:
        files = sorted(os.listdir(path + folder))
        print("Folder " + folder + " - " + str(len(files)) + " files found")
        classes.append(folder)

        for f in files:
            label.append(len(classes)-1)
            imagePath = os.path.join(path, folder, f)
            img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            data.append(img)
            #print("loading: "+ path + folder + f)

    return np.array(data), np.array(label), np.array(classes)

def splitValidation(data, label, percentValidation):
    vl = int(len(data) * percentValidation/100)
    p = np.random.permutation(len(data))

    trainData = data[p[vl:len(data)]]
    trainLabel = label[p[vl:len(data)]]
    validationData = data[p[0:vl]]
    validationLabel = label[p[0:vl]]

    return trainData, trainLabel, validationData, validationLabel

def loadTestData(path):
    data = []
    names = []

    files = sorted(os.listdir(path))
    #print(str(len(files)) + " files found")
    for f in files:
        names.append(f)
        imagePath = os.path.join(path, f)
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        data.append(img)

    return np.array(data), np.array(names)

def central_scale_images(X_imgs):
    scales = [ random.randint(5, 9)/10 ]
    print(scales)
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([X_imgs.shape[1], X_imgs.shape[2]], dtype = np.int32)
    
    X_scale_data = []
    # tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, X_imgs.shape[1], X_imgs.shape[2], 1))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data


def rotate_images(images):
    angles = []
    for i in range(len(images)):
        sinal = -1
        if (i+1)%2 == 0:
            sinal = 1

        angles.append(sinal*random.random()*np.pi/4)

    rotate = tf.contrib.image.rotate(images, angles=angles)
    # graph = tf.Graph()
    with tf.Session() as sess:
        rotated_images = sess.run(rotate)
        return rotated_images


def translate_images(images):
    translations = []
    h, w = images.shape[1:3]
    # print("w: {} h:{}".format(w, h))
    for i in range(len(images)):
        sinal = -1
        if ((i+1)%2 == 0):
            sinal = 1

        translations.append([ sinal*random.randint(0, int(h/4)), sinal*random.randint(0, int(w/4)) ])

    translate = tf.contrib.image.translate(images, translations)
    with tf.Session() as sess:
        translated_images = sess.run(translate)
        return translated_images

def resize(data, width, heigth):
    resized = []
    for i in range(len(data)):
        resized[i] = cv2.resize(data[i], (width, heigth))
    
    return np.array(resized)