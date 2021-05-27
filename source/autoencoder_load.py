# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:47:34 2021

@author: alex
"""

import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
from unet_manual import IrisImageDatabase
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

tf.debugging.set_log_device_placement(False)

img_size = (240, 320)
num_classes = 224
batch_size = 1

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
epochs = 40
inChannel = 1

input_img = Input(shape = (img_size[0], img_size[1], inChannel), name = 'input_3')

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'conv1')(input_img) #320 x 240 x 32
conv1 = BatchNormalization(name = 'batch_normalization_28')(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'conv2d_30')(conv1)
conv1 = BatchNormalization(name = 'batch_normalization_29')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_4')(conv1) #14 x 14 x 32
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'conv2d_31')(pool1) #160 x 120 x 64
conv2 = BatchNormalization(name = 'batch_normalization_30')(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'conv2d_32')(conv2)
conv2 = BatchNormalization(name = 'batch_normalization_31')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_5')(conv2) #7 x 7 x 64
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name = 'conv2d_33')(pool2) #80 x 60 x 128 (small and thick)
conv3 = BatchNormalization(name = 'batch_normalization_32')(conv3)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name = 'conv2d_34')(conv3)
conv3 = BatchNormalization(name = 'batch_normalization_33')(conv3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name = 'conv2d_35')(conv3) #80 x 60 x 256 (small and thick)
conv4 = BatchNormalization(name = 'batch_normalization_34')(conv4)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name = 'conv2d_36')(conv4)
conv4 = BatchNormalization(name = 'batch_normalization_35')(conv4)

conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', name = 'conv2d_37')(conv4) #80 x 60 x 128
conv5 = BatchNormalization(name = 'batch_normalization_36')(conv5)
conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', name = 'conv2d_38')(conv5)
conv5 = BatchNormalization(name = 'batch_normalization_37')(conv5)
conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'conv2d_39')(conv5) #80 x 60 x 64
conv6 = BatchNormalization(name = 'batch_normalization_38')(conv6)
conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'conv2d_40')(conv6)
conv6 = BatchNormalization(name = 'batch_normalization_39')(conv6)
up1 = UpSampling2D((2,2), name = 'up_sampling2d_4')(conv6) #14 x 14 x 64
conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'conv2d_41')(up1) # 160 x 120 x 32
conv7 = BatchNormalization(name = 'batch_normalization_40')(conv7)
conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'conv2d_42')(conv7)
conv7 = BatchNormalization(name = 'batch_normalization_41')(conv7)
up2 = UpSampling2D((2,2), name = 'up_sampling2d_5')(conv7) # 28 x 28 x 32
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = 'conv2d_43')(up2) # 320 x 240 x 1

# def autoencode(autoencoder_input, batch_size):
#     model = load_model('autoencoder.h5')
#     model.compile(optimizer="adam", loss="mse")
#     fluffy_seg = model.predict(
#         autoencoder_input, 
#         batch_size=batch_size,
#         verbose=2, 
#         steps=None, 
#         callbacks=None, 
#         max_queue_size=10,
#         workers=2, 
#         use_multiprocessing=False)
#     return fluffy_seg

def unet_seg(unet_input, batch_size):
    model = load_model('iris_unet.h5')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics = ['accuracy'])
    fluffy_seg = model.predict(
        unet_input, 
        batch_size=batch_size,
        verbose=2, 
        steps=None, 
        callbacks=None, 
        max_queue_size=10,
        workers=2, 
        use_multiprocessing=False)
    return fluffy_seg

def encode(autoencoder_input, batch_size):
    encoder = Model(input_img, conv4)
    encoder.load_weights('autoencoder_weights.h5', by_name = True)
    encoder.compile(optimizer="adam", loss="mse", metrics = ['loss'])
    fluffy_seg = encoder.predict(
        autoencoder_input, 
        batch_size=batch_size,
        verbose=2)
    return fluffy_seg

#%% ENCODER PREDICTION & SAVING VEATURES TO CSV
csvPath = os.path.sep.join([project_dir, "deep_features.csv"])
# Sometimes you need to treat prediction strips like a tuple
prediction_strips = GetPredInput(strip_img_paths, batch_size, img_size, return_labels = False)
# features = autoencode(prediction_strips_tuple[0], batch_size = batch_size)
features = []
features_index = []
# for i in range(prediction_strips.__len__()):
# # im_from_gen = prediction_strips.__getitem__(i)[0] # getting og image
#     features.append(encode(prediction_strips.__getitem__(i), batch_size)) #getting seg image
#     features_index.append(prediction_strips.get_index(i))
#     gc.collect()
    
csv = open(csvPath, "w")
for i in range(prediction_strips.__len__()):
    csv.write("{},{}\n".format(prediction_strips.get_index(i), encode(prediction_strips.__getitem__(i), batch_size)))
    # gc.collect()
    
csv.close()





