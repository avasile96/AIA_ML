# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:36:53 2021
This script uses resnet 50 to extract deep features from strip images
@author: alex
"""

import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from unet_import import unet_seg, polar_transform
from unet_manual import IrisImageDatabase
from skimage.color import gray2rgb
import random
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing.image import load_img

tf.debugging.set_log_device_placement(False)


img_size = (240, 320)
num_classes = 224
batch_size = 32

source_dir = os.path.dirname(os.path.abspath(__name__))
project_dir = os.path.dirname(source_dir)
dataset_dir = os.path.join(project_dir, 'dataset')
strip_folder = os.path.join(os.path.dirname(project_dir), 'strips')


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

strip_img_paths = [
        os.path.join(strip_folder, fname)
        for fname in os.listdir(strip_folder)]

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Dense, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model

latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      Flatten(),
      Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      Dense(76800, activation='sigmoid'),
      Reshape(img_size)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class IrisImageDatabase(keras.utils.Sequence):
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
        y = x
        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path, as_gray = True)
            x[j] = img
            y[j] = img
        return x, y
    
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
        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path, as_gray = True)
            x[j] = img
        return x

def GetPredInput(input_img_paths, batch_size, img_size, return_labels = False):
    # random.Random(1337).shuffle(input_img_paths)
    input_img_paths = input_img_paths
    input_gen = PredictionData(batch_size, img_size, input_img_paths)
    if (return_labels == True):
        idx = []
        idx.append(input_img_paths[26:29])
        return input_gen, idx
    else:
        return input_gen

    
# Creating strip generators (strip images have to be scaled beforehand)
train_strips = IrisImageDatabase(batch_size, img_size, strip_img_paths)


#%% AUTOENCODER
epochs = 40
inChannel = 1
input_img = Input(shape = (img_size[0], img_size[1], inChannel), name = 'input_3')

conv1 = Conv2D(8, (3, 3), activation='relu', padding='same', name = 'conv1')(input_img) #320 x 240 x 16
conv1 = BatchNormalization(name = 'batch_normalization_28')(conv1)
conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = 'conv2d_30')(conv1)
conv1 = BatchNormalization(name = 'batch_normalization_29')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_4')(conv1) #14 x 14 x 16
conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'conv2d_31')(pool1) #160 x 120 x 32
conv2 = BatchNormalization(name = 'batch_normalization_30')(conv2)
conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'conv2d_32')(conv2)
conv2 = BatchNormalization(name = 'batch_normalization_31')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_5')(conv2) #7 x 7 x 32
conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'conv2d_33')(pool2) #80 x 60 x 64 (small and thick)
conv3 = BatchNormalization(name = 'batch_normalization_32')(conv3)
conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'conv2d_34')(conv3)
conv3 = BatchNormalization(name = 'batch_normalization_33')(conv3)
conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', name = 'conv2d_35')(conv3) #80 x 60 x 128 (small and thick)
conv4 = BatchNormalization(name = 'batch_normalization_34')(conv4)
conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', name = 'conv2d_36')(conv4)
conv4 = BatchNormalization(name = 'batch_normalization_35')(conv4)
conv4 = Reshape((-1,1))(conv4)

encoded = Dense(units = 1, activation = 'relu')(conv4)

decode = Reshape((60, 80, 128))(encoded)
conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'conv2d_37')(decode) #80 x 60 x 64
conv5 = BatchNormalization(name = 'batch_normalization_36')(conv5)
conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'conv2d_38')(conv5)
conv5 = BatchNormalization(name = 'batch_normalization_37')(conv5)
conv6 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'conv2d_39')(conv5) #80 x 60 x 32
conv6 = BatchNormalization(name = 'batch_normalization_38')(conv6)
conv6 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'conv2d_40')(conv6)
conv6 = BatchNormalization(name = 'batch_normalization_39')(conv6)
up1 = UpSampling2D((2,2), name = 'up_sampling2d_4')(conv6) #14 x 14 x 64
conv7 = Conv2D(16, (3, 3), activation='relu', padding='same', name = 'conv2d_41')(up1) # 160 x 120 x 16
conv7 = BatchNormalization(name = 'batch_normalization_40')(conv7)
conv7 = Conv2D(16, (3, 3), activation='relu', padding='same', name = 'conv2d_42')(conv7)
conv7 = BatchNormalization(name = 'batch_normalization_41')(conv7)
up2 = UpSampling2D((2,2), name = 'up_sampling2d_5')(conv7) # 28 x 28 x 16
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = 'conv2d_43')(up2) # 320 x 240 x 1


# define autoencoder model
autoencoder = Model(input_img, decoded)
# compile autoencoder model
autoencoder.compile(optimizer='adam', loss='mse')
# set checkpoints
checkpoint_filepath = 'autoencoder_weights.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='loss',
    save_best_only=True)
# fit the autoencoder model to reconstruct input
history = autoencoder.fit(train_strips, batch_size=batch_size, epochs=epochs, callbacks = [],
                          verbose=1)
features = autoencoder.predict(train_strips.__getitem__(0)[0])
# Model Save
# autoencoder.save(
#     'autoencoder.h5', overwrite=True, include_optimizer=True, save_format=None,
#     signatures=None, options=None, save_traces=True)






