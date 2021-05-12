# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:41:36 2021

@author: alex
"""

import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from unet_import import GetInputGenerator
from skimage.color import gray2rgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.debugging.set_log_device_placement(True)


img_size = (240, 320)
num_classes = 2
batch_size = 1

source_dir = os.path.dirname(os.path.abspath(__name__))
project_dir = os.path.dirname(source_dir)
big_file_dir = os.path.dirname(project_dir)
dataset_dir = os.path.join(project_dir, 'dataset')
strip_folder = os.path.join(source_dir, 'strips')


input_img_paths = []
for patient_index in os.listdir(os.path.join(dataset_dir, 'images')):
    if os.path.isdir(os.path.join(dataset_dir, 'images', patient_index)):
        # patient_dir = os.path.join(dataset_dir, 'images', patient_index)
        for fname in os.listdir(os.path.join(dataset_dir, 'images', patient_index)):
            if fname.endswith(".bmp") and not fname.startswith("."):
                input_img_paths.append(os.path.join(dataset_dir, 'images', patient_index, fname))

target_img_paths = [
        os.path.join(dataset_dir, 'groundtruth', fname)
        for fname in os.listdir(os.path.join(dataset_dir, 'groundtruth'))
        if fname.endswith(".tiff") and not fname.startswith(".")]


if __name__ == '__main__':
    
    # Reading from csv
    db = pd.read_csv(big_file_dir+"\\features.csv")
    db_np = db.to_numpy()
    
    # Getting labels and features
    labels = db_np[:,0]
    classes = np.unique(labels)
    nr_of_classes = classes.shape[0]
    features = db_np[:,1:]
    
    # Scaling
    features_scaled = MinMaxScaler().fit_transform(features)
    
    #%% Abit of data exploration
    nonzero_element_count = np.count_nonzero(features_scaled)
    zero_element_count = features_scaled.size - np.count_nonzero(features_scaled)
    
    #correlation plot
    corr = db.corr()
    f, ax = plt.subplots(figsize=(30, 30))
    sns.heatmap(corr,annot=True)
    
    # Generator from csv
    
    #%% Model
    # define our simple neural network
    """
    How did I come up with the values of 256 and 16 for the two hidden layers?
    A good rule of thumb is to take the square root of the previous number of nodes 
    in the layer and then find the closest power of 2.
    In this case, the closest power of 2 to 100352 is 256. The square root of 256
    is then 16, thus giving us our architecture definition.
    """
    # model = Sequential()
    # model.add(Dense(256, input_shape=(7 * 7 * 2048,), activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(Dense(16, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(Dense(nr_of_classes))
    model = tf.keras.models.Sequential([
              tf.keras.layers.Flatten(input_shape=(7 * 7 * 2048,)),
              tf.keras.layers.Dense(512, activation='relu'),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(nr_of_classes)
              ])
    # compile the model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss_fn, optimizer='adam',
    	metrics=["accuracy"])
    # Train 
    history = model.fit(features_scaled, labels, epochs=20)
    # Evaluate
    model.evaluate(features_scaled,  labels, verbose=2)    

    
    