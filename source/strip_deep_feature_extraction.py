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

# def encoder(input_img):
#     #encoder
#     #input = 320 x 240 x 1 (wide and thin)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
#     conv1 = BatchNormalization()(conv1)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
#     conv1 = BatchNormalization()(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
#     conv2 = BatchNormalization()(conv2)
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
#     conv2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
#     conv3 = BatchNormalization()(conv3)
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#     conv3 = BatchNormalization()(conv3)
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
#     conv4 = BatchNormalization()(conv4)
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#     conv4 = BatchNormalization()(conv4)
#     return conv4

# def decoder(conv4):    
#     #decoder
#     conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
#     conv5 = BatchNormalization()(conv5)
#     conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
#     conv5 = BatchNormalization()(conv5)
#     conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
#     conv6 = BatchNormalization()(conv6)
#     conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
#     conv6 = BatchNormalization()(conv6)
#     up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
#     conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
#     conv7 = BatchNormalization()(conv7)
#     conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
#     conv7 = BatchNormalization()(conv7)
#     up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
#     decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
#     return decoded




    
# Creating strip generators (strip images have to be scaled beforehand)
train_strips = IrisImageDatabase(batch_size, img_size, input_img_paths, input_img_paths)

#%% AUTOENCODER
epochs = 20
inChannel = 1
input_img = Input(shape = (img_size[0], img_size[1], inChannel))

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
conv3 = BatchNormalization()(conv3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
conv4 = BatchNormalization()(conv4)

conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
conv5 = BatchNormalization()(conv5)
conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
conv6 = BatchNormalization()(conv6)
up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
conv7 = BatchNormalization()(conv7)
up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1


# define autoencoder model
autoencoder = Model(input_img, decoded)
# compile autoencoder model
autoencoder.compile(optimizer='adam', loss='mse')
# plot the autoencoder
plot_model(autoencoder, 'autoencoder_compress.png', show_shapes=True)
# fit the autoencoder model to reconstruct input
history = autoencoder.fit(train_strips, batch_size=batch_size, epochs=epochs, verbose=1)
# Encoder prediction
encoder = Model(input_img, conv4)
features = encoder.predict(train_strips)
print("[INFO] loading network...")

#%% SAVING VEATURES TO CSV
csvPath = os.path.sep.join([project_dir, "features.csv"])
csv = open(csvPath, "w")

for (label, vec) in zip(labels, features_mod):
    # construct a row that exists of the class label and
    # extracted features
    vec = ",".join([str(v) for v in vec])
    csv.write("{},{}\n".format(label, vec))
    
csv.close()

del features
del x_resized
gc.collect()




# if __name__ == '__main__':
    
#     # Creating strip generators (strip images have to be scaled beforehand)
#     train_strips = IrisImageDatabase(batch_size, img_size, input_img_paths, input_img_paths)

#     #%% AUTOENCODER
#     epochs = 20
#     inChannel = 1
#     input_img = Input(shape = (img_size[0], img_size[1], inChannel))
    
    
#     # define autoencoder model
#     autoencoder = Model(input_img, decoder(encoder(input_img)))
#     # compile autoencoder model
#     autoencoder.compile(optimizer='adam', loss='mse')
#     # plot the autoencoder
#     plot_model(autoencoder, 'autoencoder_compress.png', show_shapes=True)
#     # fit the autoencoder model to reconstruct input
#     history = autoencoder.fit(train_strips, batch_size=batch_size, epochs=epochs, verbose=1)
#     #
#     features = encoder.predict(train_strips)
#     print("[INFO] loading network...")
    
#     #%% SAVING VEATURES TO CSV
#     csvPath = os.path.sep.join([project_dir, "features.csv"])
#     csv = open(csvPath, "w")
    
#     for (label, vec) in zip(labels, features_mod):
#         # construct a row that exists of the class label and
#         # extracted features
#         vec = ",".join([str(v) for v in vec])
#         csv.write("{},{}\n".format(label, vec))
        
#     csv.close()
    
#     del features
#     del x_resized
#     gc.collect()
    

