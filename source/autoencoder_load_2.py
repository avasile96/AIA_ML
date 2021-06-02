# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:47:34 2021

@author: alex
"""

import os
import numpy as np
from skimage import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.layers import Conv2D
from autoencoder_training_medium import create_cbcb, create_pooling, create_upsamping
import pandas as pd

tf.debugging.set_log_device_placement(False)

img_size = (240, 320)
num_classes = 224
batch_size = 32

source_dir = os.path.dirname(os.path.abspath(__name__))
project_dir = os.path.dirname(source_dir)
dataset_dir = os.path.join(project_dir, 'dataset')
strip_folder = os.path.join(os.path.dirname(project_dir), 'strips')

strip_img_paths = [
        os.path.join(strip_folder, fname)
        for fname in os.listdir(strip_folder)]

latent_dim = 64 
    
class PredictionData(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        idx = []
        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path, as_gray = True)
            x[j] = img
        return x
    def get_index(self, idx):
            """Returns tuple (input, target) correspond to batch #idx."""
            i = idx * self.batch_size
            batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
            index = []
            for j, path in enumerate(batch_input_img_paths):
                index.append(batch_input_img_paths[j][26:29])
            return index

def GetPredInput(input_img_paths, batch_size, img_size, return_labels = False):
    # random.Random(1337).shuffle(input_img_paths)
    input_img_paths = input_img_paths
    input_gen = PredictionData(batch_size, img_size, input_img_paths)
    return input_gen


#%% AUTOENCODER
epochs = 60

def encode(autoencoder_input, batch_size):
    
    init_mode = "he_normal"
    inChannel = 1
    f = 4 # number of starting filters
    naming_factor = 1 # counter for control over layer names
    
    input_img = Input(shape = (img_size[0], img_size[1], inChannel), name = 'input_3')
    
    conv1, naming_factor = create_cbcb(input_img, f, init_mode, naming_factor)
    pool1 = create_pooling(conv1, naming_factor)
    conv2, naming_factor = create_cbcb(pool1, f*2, init_mode, naming_factor)
    conv3, naming_factor = create_cbcb(conv2, f*2, init_mode, naming_factor)
    pool3 = create_pooling(conv3, naming_factor)
    conv3, naming_factor = create_cbcb(pool3, f*2, init_mode, naming_factor)
    pool4 = create_pooling(conv3, naming_factor)
    ########################################################
    conv5, naming_factor = create_cbcb(pool4, f*2, init_mode, naming_factor)
    ########################################################
    conv5_res = Reshape((-1,1))(conv5) # Feature Layer
    
    encoder = Model(input_img, conv5_res)
    encoder.load_weights('autoencoder_medium_weights_f_{}.h5'.format(f), by_name = True)
    encoder.compile(optimizer="adam", loss="mse", metrics = ['loss'])
    feat_vect = encoder.predict(
        autoencoder_input, 
        batch_size=batch_size,
        verbose=2)
    return feat_vect

def strip_pred(prediction_strips, feat_vect, csvPath):
    # Pandas approach
    """
    Obs: there is now difference between this approach and the one in
    "Fill feature vector one prediciton at a time"
    from an index - instance point of view
    """
    
    features_index = []
    for i in range(prediction_strips.__len__()):
        # Extend functions like append but for lists, not elements
        # Useful when dealing with batches
        features_index.extend(prediction_strips.get_index(i))
    
    df = pd.DataFrame(data = features_d, index = features_index)
    
    df.to_csv(csvPath)
    

#%% ENCODER PREDICTION & SAVING VEATURES TO CSV
if __name__ == '__main__':
    
    csvPath = os.path.sep.join([project_dir, "deep_features.csv"])

    prediction_strips = GetPredInput(strip_img_paths, batch_size, img_size, return_labels = False)
    
    features_d = np.squeeze(encode(prediction_strips, batch_size))
    
    strip_pred(prediction_strips, features_d, csvPath)





