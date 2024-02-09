import numpy as np
import os
import scipy.io
import struct
from array import array

file_path = os.path.dirname(__file__)


##### LOAD ALPHA DIGITS #####
datafile_alpha_digits = scipy.io.loadmat(f'{file_path}/../data/binaryalphadigs.mat')

def lire_alpha_digits(characters_to_read, datafile=datafile_alpha_digits, indices=None):
    def character_index(c):
        if c.isnumeric():
            return int(c)
        return ord(c) - ord('A') + 10

    if indices is None:
        indices = [character_index(c) for c in characters_to_read]
    
    X = np.array([datafile['dat'][indices[0]][i].flatten() for i in range(len(datafile['dat'][indices[0]]))])
    for index in indices[1:]:
        X_i = np.array([datafile['dat'][index][i].flatten() for i in range(len(datafile['dat'][index]))])
        X = np.concatenate([X, X_i])
    return X


##### LOAD MNIST #####
train_images_filepath = f'{file_path}/../data/mnist/train-images.idx3-ubyte'
train_labels_filepath = f'{file_path}/../data/mnist/train-labels.idx1-ubyte'
test_images_filepath = f'{file_path}/../data/mnist/t10k-images.idx3-ubyte'
test_labels_filepath = f'{file_path}/../data/mnist/t10k-labels.idx1-ubyte'

def read_images_labels(images_filepath, labels_filepath):        
    labels = []
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())        
    
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img            
    
    return np.array(images), np.array(labels)

def one_hot_encode(y):
    num_classes = len(np.unique(y))
    one_hot_labels = np.zeros((y.shape[0], num_classes))
    one_hot_labels[np.arange(y.shape[0]), y] = 1
    return one_hot_labels

def load_mnist(train_images_filepath=train_images_filepath, train_labels_filepath=train_labels_filepath,
               test_images_filepath=test_images_filepath, test_labels_filepath=test_labels_filepath):
    X_train, y_train = read_images_labels(train_images_filepath, train_labels_filepath)
    X_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath)
    
    X_train = ((X_train / 255.) > 0.5).reshape(-1, 28*28)
    X_test = ((X_test / 255.) > 0.5).reshape(-1 ,28*28)
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    
    return X_train.astype(int), y_train, X_test.astype(int), y_test