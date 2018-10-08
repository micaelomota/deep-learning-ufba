import os
import numpy as np
import cv2
import os

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
