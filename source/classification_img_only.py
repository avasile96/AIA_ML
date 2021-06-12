# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:03:27 2021

@author: alex
"""

import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from unet_manual import create_patients


tf.debugging.set_log_device_placement(True)


img_size = (240, 320)
num_classes = 224
batch_size = 64

source_dir = os.path.dirname(os.path.abspath(__name__))
project_dir = os.path.dirname(source_dir)
big_file_dir = os.path.dirname(project_dir)
strip_folder = os.path.join(big_file_dir, 'strips')
db_path = project_dir+"\\deep_features.csv"


def get_strips(strip_folder):
    strips = []
    pat_indices = []
    for strip_file in os.listdir(strip_folder):
        pat_indices.append(strip_file[8:11])
        strips.append(io.imread(os.path.join(strip_folder, strip_file)))
        
    return strips, pat_indices

def define_model():
    model = Sequential()
    model.add(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(320,240,1)))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(225, activation='softmax', kernel_initializer='he_uniform'))
    return model

if __name__ == '__main__':
    
    #%% Data Reading
    strips, indices = get_strips(strip_folder)
    
    strips = np.array(strips, dtype = np.float)
    frame = np.zeros((2240,240,320,1))
    frame[:,:,:,0] = strips

    train_x = frame
    train_y = to_categorical(np.array(indices), 225)
    
    del strips, frame, indices
    gc.collect()
    
    #%% 50-50 Split
    x_train = []
    y_train = []
   
    x_val = [train_x[0]]
    y_val = [train_y[5]]
    
    for i in range(0,2250,10):
        print(i)
        x_train.extend(train_x[i:i+5])
        y_train.extend(train_y[i:i+5])
        
        x_val.extend(train_x[i+6:i+11])
        y_val.extend(train_y[i+6:i+11])
        
    x_train = np.array(x_train, dtype = np.float)
    x_val = np.array(x_val, dtype = np.float)
    
    y_train = np.array(y_train, dtype = np.int)
    y_val = np.array(y_val, dtype = np.int)
    
    np.random.seed(42)
    np.random.shuffle(x_train)
    
    np.random.seed(42)
    np.random.shuffle(y_train)
    
    np.random.seed(69)
    np.random.shuffle(x_val)
    
    np.random.seed(69)
    np.random.shuffle(y_val)
    
    plt.figure()
    io.imshow(y_train)
    plt.title('y_train')
    
    plt.figure()
    io.imshow(y_val)
    plt.title('y_val')
    
    del train_x, train_y
    gc.collect()
    
    #%% Shallow neural net
    """
    How did I come up with the values of 256 and 16 for the two hidden layers?
    A good rule of thumb is to take the square root of the previous number of nodes 
    in the layer and then find the closest power of 2.
    In this case, the closest power of 2 to 100352 is 256. The square root of 256
    is then 16, thus giving us our architecture definition.
    """
    model = define_model()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    model.compile(loss=loss_fn, optimizer='adam', metrics=["accuracy"])
    n_ep = 10
    # Train 
    history = model.fit(x=x_train, y=y_train, validation_data=(x_val,y_val), epochs=n_ep, verbose = 2, batch_size = batch_size)
    
    a = model.predict(x_val)
    
    #%% Plotting
    y_ax = np.linspace(0,100,len(history.history["accuracy"]), dtype = np.int)
    x_ax = np.linspace(0,n_ep,len(history.history["accuracy"]), dtype = np.int)
    
    plt.figure()
    plt.plot(x_ax,np.array(history.history["loss"]))
    plt.xlabel('epochs')
    plt.title('Shallow Conv Net Classification Loss')
    
    plt.figure()
    plt.plot(x_ax,np.array(history.history["accuracy"])*100)
    plt.xlabel('epochs')
    plt.ylabel('[%]')
    plt.title('Shallow Conv Net Classification Accuracy')
    
    