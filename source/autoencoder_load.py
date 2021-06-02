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
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D, Reshape
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

def create_encoder(init_mode = 'he_normal'):
    inChannel = 1
    input_img = Input(shape = (img_size[0], img_size[1], inChannel), name = 'input_3')
    f = 2
    
    conv1 = Conv2D(f, (3, 3), activation='relu', padding='same', name = 'conv1', 
                   kernel_initializer = init_mode)(input_img) #320 x 240 x f
    conv1 = BatchNormalization(name = 'batch_normalization_28')(conv1)
    conv1 = Conv2D(f*2, (3, 3), activation='relu', padding='same', name = 'conv2d_30',
                   kernel_initializer = init_mode)(conv1)
    conv1 = BatchNormalization(name = 'batch_normalization_29')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_4')(conv1) #160 x 120 x 2f
    conv2 = Conv2D(f*4, (3, 3), activation='relu', padding='same', name = 'conv2d_31',
                   kernel_initializer = init_mode)(pool1) #160 x 120 x 4f
    conv2 = BatchNormalization(name = 'batch_normalization_30')(conv2)
    conv2 = Conv2D(f*4, (3, 3), activation='relu', padding='same', name = 'conv2d_32',
                   kernel_initializer = init_mode)(conv2)
    conv2 = BatchNormalization(name = 'batch_normalization_31')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_5')(conv2) #80 x 60 x 4f
    conv3 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_33',
                   kernel_initializer = init_mode)(pool2) #80 x 60 x 8f (small and thick)
    conv3 = BatchNormalization(name = 'batch_normalization_32')(conv3)
    conv3 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_34',
                   kernel_initializer = init_mode)(conv3)
    conv3 = BatchNormalization(name = 'batch_normalization_33')(conv3)
    conv4 = Conv2D(f*16, (3, 3), activation='relu', padding='same', name = 'conv2d_35',
                   kernel_initializer = init_mode)(conv3) #80 x 60 x 16f (small and thick)
    conv4 = BatchNormalization(name = 'batch_normalization_34')(conv4)
    conv4 = Conv2D(f*16, (3, 3), activation='relu', padding='same', name = 'conv2d_36',
                   kernel_initializer = init_mode)(conv4)
    conv4 = BatchNormalization(name = 'batch_normalization_35')(conv4)
    ########################################################
    conv5_res = Reshape((-1,1))(conv4) # Feature Layer
    conv5_shape = tuple(conv4.shape)    
    conv6 = Reshape((conv5_shape[1], conv5_shape[2], conv5_shape[3]))(conv5_res)
    ########################################################
    conv7 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_41',
                    kernel_initializer = init_mode)(conv6) #80 x 60 x 8f
    conv7 = BatchNormalization(name = 'batch_normalization_40')(conv7)
    conv7 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_42',
                    kernel_initializer = init_mode)(conv7)
    conv7 = BatchNormalization(name = 'batch_normalization_41')(conv7)
    conv8 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_43',
                   kernel_initializer = init_mode)(conv7) #80 x 60 x 4f
    conv8 = BatchNormalization(name = 'batch_normalization_42')(conv8)
    conv8 = Conv2D(f*4, (3, 3), activation='relu', padding='same', name = 'conv2d_44',
                   kernel_initializer = init_mode)(conv8)
    conv8 = BatchNormalization(name = 'batch_normalization_43')(conv8)
    up2 = UpSampling2D((2,2), name = 'up_sampling2d_5')(conv8) #160 x 120 x 2f
    conv9 = Conv2D(f*2, (3, 3), activation='relu', padding='same', name = 'conv2d_45',
                   kernel_initializer = init_mode)(up2) # 160 x 120 x 2f
    conv9 = BatchNormalization(name = 'batch_normalization_44')(conv9)
    conv9 = Conv2D(f, (3, 3), activation='relu', padding='same', name = 'conv2d_46',
                   kernel_initializer = init_mode)(conv9)
    conv9 = BatchNormalization(name = 'batch_normalization_45')(conv9)
    up3 = UpSampling2D((2,2), name = 'up_sampling2d_6')(conv9) # 320 x 240 x f
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = 'conv2d_47',
                     kernel_initializer = init_mode)(up3) # 320 x 240 x 1
    # define autoencoder model
    encoder = Model(input_img, conv5_res)
    encoder.load_weights('autoencoder_weights.h5', by_name = True)
    encoder.compile(optimizer='adam', loss='mse')
    return encoder

def encode(autoencoder_input, batch_size):
    encoder = create_encoder()

    feat_vect = encoder.predict(
        autoencoder_input, 
        batch_size=batch_size,
        verbose=2)
    return feat_vect

#%% ENCODER PREDICTION & SAVING VEATURES TO CSV

# Sometimes you need to treat prediction strips like a tuple
prediction_strips = GetPredInput(strip_img_paths, batch_size, img_size, return_labels = False)
features = encode(prediction_strips[0], batch_size = batch_size)
# features = []
# features_index = []
# for i in range(prediction_strips.__len__()):
# # im_from_gen = prediction_strips.__getitem__(i)[0] # getting og image
#     features.append(encode(prediction_strips.__getitem__(i), batch_size)) #getting seg image
#     features_index.append(prediction_strips.get_index(i))
#     gc.collect()
    
# csv = open(csvPath, "w")
# for i in range(prediction_strips.__len__()):
#     csv.write("{},{}\n".format(prediction_strips.get_index(i), encode(prediction_strips.__getitem__(i), batch_size)))
#     # gc.collect()
    
# csv.close()





