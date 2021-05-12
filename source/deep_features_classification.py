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
from tensorflow.keras.utils import to_categorical
import keras
import csv

tf.debugging.set_log_device_placement(True)


img_size = (240, 320)
num_classes = 2
batch_size = 10

source_dir = os.path.dirname(os.path.abspath(__name__))
project_dir = os.path.dirname(source_dir)
big_file_dir = os.path.dirname(project_dir)
dataset_dir = os.path.join(project_dir, 'dataset')
strip_folder = os.path.join(source_dir, 'strips')
# db = pd.read_csv(big_file_dir+"\\features.csv")
db_path = big_file_dir+"\\features.csv"

# # Getting labels and features
# labels = db_np[:,0]
# classes = np.unique(labels)
# nr_of_classes = classes.shape[0]
# features = db_np[:,1:]



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

class FeaturesLabelsData(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, db_path, numClasses, mode = 'train'):
        self.batch_size = batch_size
        self.db_path = db_path
        self.file = open(db_path,'r')
        self.reader = csv.reader(self.file)
        self.mode = mode
        self.numClasses = numClasses
        
    def __len__(self):
        return len(list(self.reader)) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        while True:
            # initialize our batch of data and labels
            data = []
            labels = []
            # keep looping until we reach our batch size
            while len(data) < self.batch_size:
                # attempt to read the next row of the CSV file
                row = self.file.readline()
                # check to see if the row is empty, indicating we have
                # reached the end of the file
                if row == "":
                    # reset the file pointer to the beginning of the file
                    # and re-read the row
                    self.file.seek(0)
                    row = self.file.readline()
                    # if we are evaluating we should now break from our
                    # loop to ensure we don't continue to fill up the
                    # batch from samples at the beginning of the file
                    if self.mode == "eval":
                        break
                # extract the class label and features from the row
                row = row.strip().split(",")
                label = row[0]
                label = to_categorical(label, num_classes=self.numClasses)
                features = np.array(row[1:], dtype="float")
                # update the data and label lists
                data.append(features)
                labels.append(label)
                # yield the batch to the calling function
            return (np.array(data), np.array(labels))

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
    
    # Generator from csv
    train_gen = FeaturesLabelsData(batch_size, db_path, nr_of_classes, mode = 'train')
    del db, db_np, features
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
    history = model.fit(train_gen, epochs=20)
    # Evaluate
    # model.evaluate(features_scaled,  labels, verbose=2)    

    
    