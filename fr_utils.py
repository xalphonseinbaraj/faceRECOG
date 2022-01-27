import tensorflow as tf
import numpy as np
import os
import cv2
from numpy import genfromtxt
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import h5py
import matplotlib.pyplot as plt

def img_path_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1, model)
    

def img_to_encoding(image, model):
    image = cv2.resize(image, (96, 96))
    # image = cv2.resize(image, (224, 224)) #comment if use any pretrained model
    img = image[...,::-1]  #comment if use any pretrained model
    img = np.around(np.transpose(img,(2,0,1))/255.0, decimals=12)#comment if use any pretrained model
    train = np.array([img])#comment if use any pretrained model
    # train = np.array([image])#comment if use any pretrained model
    embedding = model.predict(train)
    return embedding