import os
import numpy as np
import cv2
import tensorflow as tf
from math import ceil, floor
import uuid 

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
    
    #validationItems = np.random.choice(data, validationLength, replace=False)
    #fData = removeAll(validationItems, data)
    #print("fData size: " + str(len(fData)))

    #d = dict(label=folder, train=fData, validation=validationItems)
    #data.append(d)

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


# aumentation saving images inside the same path
def aumentation(path):
    print("Running aumentation - scale and translate images")
    IMAGE_HEIGHT = 77
    IMAGE_WIDTH = 71
    NUM_CHANNELS = 1
    folders = sorted(os.listdir(path))
    
    for folder in folders:
        files = sorted(os.listdir(path + folder))
        print("Folder " + folder + " - " + str(len(files)) + " files found")

        data = []

        for f in files:
            imagePath = os.path.join(path, folder, f)
            img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            data.append(img)

        data = np.array(data)
        data = np.reshape(data, (len(data), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))


        scaledImages = central_scale_images(data, [0.90, 0.75, 0.60])

        for img in scaledImages:
            imgName = uuid.uuid4().hex[:8].upper() + ".png"
            saveImagePath = os.path.join(path, folder, imgName)
            cv2.imwrite(saveImagePath, img)
            print(saveImagePath)

        translated = translate_images(data)

        for img in translated:
            imgName = uuid.uuid4().hex[:8].upper() + ".png"
            saveImagePath = os.path.join(path, folder, imgName)
            cv2.imwrite(saveImagePath, img)
            print(saveImagePath)


def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([X_imgs.shape[1], X_imgs.shape[2]], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
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



def get_translate_parameters(index, IMAGE_SIZE):
    if index == 0: # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1: # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2: # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE)) 
    else: # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE 
        
    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype = np.float32)
    n_translations = 4
    X_translated_arr = []
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), X_imgs.shape[1], X_imgs.shape[2], 1), dtype = np.float32)
            X_translated.fill(0.0) # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i, X_imgs.shape[2])
            offsets[:, :] = base_offset 
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)
            
            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], \
			 w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)

    X_translated_arr = np.array(X_translated_arr, dtype = np.float32)
    return X_translated_arr