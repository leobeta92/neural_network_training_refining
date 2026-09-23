import tensorflow as tf
from tensorflow import keras as ks
from matplotlib import pyplot as plt
import numpy as np
import time
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

def unpickle(file):
    # import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(data_dir):
#load training data
 for i in range(1, 6):
  filename = data_dir+"data_batch_"+str(i)
  dictionary = unpickle(filename)
  x_data = dictionary[b'data']
  y_data = np.array(dictionary[b"labels"])
  if i==1:
   x_train = x_data
   y_train= y_data
  else:
   x_train = np.concatenate((x_train, x_data), axis = 0)
   y_train = np.concatenate((y_train, y_data), axis = 0)
#load testing data
 filename = data_dir+"test_batch"
 dictionary = unpickle(filename)
 x_test = dictionary[b"data"]
 y_test = np.array(dictionary[b"labels"])
 return x_train, y_train, x_test, y_test

def parse_record(record):
 depth_major = record.reshape((3, 32, 32))
 image = np.transpose(depth_major, [1, 2, 0])
 return image

