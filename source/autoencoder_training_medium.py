# -*- coding: utf-8 -*-
"""

Created on Mon May 10 23:36:53 2021

This script uses a priprietary autoencoder to extract deep features from strip images.

Variations on this architecture should be found in the autoencoders dir outside repository.

@author: alex
"""

import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

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

from tensorflow.keras.models import Model, load_model
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
        y = x # (use for training autoencoder)
        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path, as_gray = True)
            x[j] = img
            y[j] = img # (use for training autoencoder)
        return x, y # (use for training autoencoder)
        # return x # use for encoder stuff?
    
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

def create_cbcb(prev_layer, f, init_mode, naming_factor):
    cbcb = Conv2D(f, (3, 3), activation='relu', padding='same', name = 'conv2d_{}'.format(naming_factor),
                   kernel_initializer = init_mode)(prev_layer) #160 x 120 x 4f
    cbcb = BatchNormalization(name = 'batch_normalization_{}'.format(naming_factor))(cbcb)
    cbcb = Conv2D(f, (3, 3), activation='relu', padding='same', name = 'conv2d_{}'.format(naming_factor+1),
                   kernel_initializer = init_mode)(cbcb)
    cbcb = BatchNormalization(name = 'batch_normalization_{}'.format(naming_factor+1))(cbcb)
    naming_factor +=2
    return cbcb, naming_factor

def create_pooling(prev_layer, naming_factor):
    poolayer = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_{}'.format(naming_factor))(prev_layer) 
    return poolayer

def create_upsamping(prev_layer, naming_factor):
    ups = UpSampling2D((2,2), name = 'up_sampling2d_{}'.format(naming_factor))(prev_layer)
    return ups


#%% AUTOENCODER
def create_autoencoder(init_mode = 'he_normal'):
    inChannel = 1
    input_img = Input(shape = (img_size[0], img_size[1], inChannel), name = 'input_3')
    f = 4
    
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
    pool3 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_6')(conv4) #40 x 30 x 16f
    ########################################################
    conv5 = Conv2D(f*16, (3, 3), activation='relu', padding='same', name = 'conv2d_37',
                   kernel_initializer = init_mode)(pool3) #160 x 120 x 16f
    conv5 = BatchNormalization(name = 'batch_normalization_36')(conv5)
    conv5 = Conv2D(f*16, (3, 3), activation='relu', padding='same', name = 'conv2d_38',
                   kernel_initializer = init_mode)(conv5)
    conv5 = BatchNormalization(name = 'batch_normalization_37')(conv5)
    ########################################################
    conv5_res = Reshape((-1,1))(conv5) # Feature Layer
    conv5_shape = tuple(conv5.shape)    
    conv6 = Reshape((conv5_shape[1], conv5_shape[2], conv5_shape[3]))(conv5_res)
    ########################################################
    conv6 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_39',
                   kernel_initializer = init_mode)(conv6) #80 x 60 x 8f
    conv6 = BatchNormalization(name = 'batch_normalization_38')(conv6)
    conv6 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_40',
                   kernel_initializer = init_mode)(conv6)
    conv6 = BatchNormalization(name = 'batch_normalization_39')(conv6)
    #########################################################
    up1 = UpSampling2D((2,2), name = 'up_sampling2d_4')(conv6) #160 x 120 x 2f
    conv7 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_41',
                    kernel_initializer = init_mode)(up1) #80 x 60 x 8f
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
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, f

def create_medium_ae(init_mode = 'he_normal'):
    inChannel = 1
    input_img = Input(shape = (img_size[0], img_size[1], inChannel), name = 'input_3')
    f = 4 # number of starting filters
    
    naming_factor = 1 # start counting convolutions from 3 because network starts with 1 and 2
    
    conv1, naming_factor = create_cbcb(input_img, f, init_mode, naming_factor)
    pool1 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_{}'.format(naming_factor))(conv1) 
    
    conv2, naming_factor = create_cbcb(pool1, f*2, init_mode, naming_factor)
    conv2, naming_factor = create_cbcb(conv2, f*2, init_mode, naming_factor)
    pool2 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_{}'.format(naming_factor))(conv2) 
    
    conv3, naming_factor = create_cbcb(pool2, f*2, init_mode, naming_factor)
    conv4, naming_factor = create_cbcb(conv3, f*2, init_mode, naming_factor)
    pool3 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_{}'.format(naming_factor))(conv4) 
    ########################################################
    conv5, naming_factor = create_cbcb(pool3, f*2, init_mode, naming_factor)
    ########################################################
    conv5_res = Reshape((-1,1))(conv5) # Feature Layer
    conv5_shape = tuple(conv5.shape)    
    conv6 = Reshape((conv5_shape[1], conv5_shape[2], conv5_shape[3]))(conv5_res)
    ########################################################
    conv6, naming_factor = create_cbcb(conv6, f*2, init_mode, naming_factor)
    #########################################################
    up1 = UpSampling2D((2,2), name = 'up_sampling2d_{}'.format(naming_factor))(conv6) 

    conv7, naming_factor = create_cbcb(up1, f*2, init_mode, naming_factor)

    conv8, naming_factor = create_cbcb(conv7, f*2, init_mode, naming_factor)
    up2 = UpSampling2D((2,2), name = 'up_sampling2d_{}'.format(naming_factor))(conv8) 

    conv9, naming_factor = create_cbcb(up2, f*2, init_mode, naming_factor)

    conv10, naming_factor = create_cbcb(conv9, f*2, init_mode, naming_factor)
    up3 = UpSampling2D((2,2), name = 'up_sampling2d_{}'.format(naming_factor))(conv10)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = 'conv2d_end',
                     kernel_initializer = init_mode)(up3) 
     # define autoencoder model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, f

def create_deep_ae(init_mode = 'he_normal'):
    inChannel = 1
    input_img = Input(shape = (img_size[0], img_size[1], inChannel), name = 'input_3')
    f = 4 # number of starting filters
    
    naming_factor = 1 # start counting convolutions from 3 because network starts with 1 and 2
    
    conv1, naming_factor = create_cbcb(input_img, f, init_mode, naming_factor)
    pool1 = create_pooling(conv1, naming_factor)
    
    conv2, naming_factor = create_cbcb(pool1, f*2, init_mode, naming_factor)
    conv2, naming_factor = create_cbcb(conv2, f*2, init_mode, naming_factor)
    conv2, naming_factor = create_cbcb(conv2, f*2, init_mode, naming_factor)
    conv2, naming_factor = create_cbcb(conv2, f*2, init_mode, naming_factor)
    pool2 = create_pooling(conv2, naming_factor)
    
    conv3, naming_factor = create_cbcb(pool2, f*2, init_mode, naming_factor)
    conv3, naming_factor = create_cbcb(conv3, f*2, init_mode, naming_factor)
    pool3 = create_pooling(conv3, naming_factor)
    conv3, naming_factor = create_cbcb(pool3, f*2, init_mode, naming_factor)
    conv3, naming_factor = create_cbcb(conv3, f*2, init_mode, naming_factor)
    pool4 = pool2 = create_pooling(conv3, naming_factor)
    ########################################################
    conv5, naming_factor = create_cbcb(pool4, f*2, init_mode, naming_factor)
    ########################################################
    conv5_res = Reshape((-1,1))(conv5) # Feature Layer
    conv5_shape = tuple(conv5.shape)    
    conv6 = Reshape((conv5_shape[1], conv5_shape[2], conv5_shape[3]))(conv5_res)
    ########################################################
    conv6, naming_factor = create_cbcb(conv6, f*2, init_mode, naming_factor)
    #########################################################
    up1 = create_upsamping(conv6, naming_factor)

    conv7, naming_factor = create_cbcb(up1, f*2, init_mode, naming_factor)
    conv7, naming_factor = create_cbcb(conv7, f*2, init_mode, naming_factor)
    up2 = create_upsamping(conv7, naming_factor)
    conv7, naming_factor = create_cbcb(up2, f*2, init_mode, naming_factor)
    conv7, naming_factor = create_cbcb(conv7, f*2, init_mode, naming_factor)

    up3 = create_upsamping(conv7, naming_factor)

    conv9, naming_factor = create_cbcb(up3, f*2, init_mode, naming_factor)
    conv9, naming_factor = create_cbcb(conv9, f*2, init_mode, naming_factor)
    conv9, naming_factor = create_cbcb(conv9, f*2, init_mode, naming_factor)
    conv9, naming_factor = create_cbcb(conv9, f*2, init_mode, naming_factor)

    conv10, naming_factor = create_cbcb(conv9, f*2, init_mode, naming_factor)
    up4 = create_upsamping(conv10, naming_factor)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = 'conv2d_end',
                     kernel_initializer = init_mode)(up4) 
     # define autoencoder model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, f

def create_custom_ae(init_mode = 'he_normal'):
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
    conv5_shape = tuple(conv5.shape)    
    conv6 = Reshape((conv5_shape[1], conv5_shape[2], conv5_shape[3]))(conv5_res)
    ########################################################
    conv6, naming_factor = create_cbcb(conv6, f*2, init_mode, naming_factor)
    #########################################################
    up1 = create_upsamping(conv6, naming_factor)
    conv7, naming_factor = create_cbcb(up1, f*2, init_mode, naming_factor)
    conv7, naming_factor = create_cbcb(conv7, f*2, init_mode, naming_factor)
    up3 = create_upsamping(conv7, naming_factor)
    conv9, naming_factor = create_cbcb(up3, f*2, init_mode, naming_factor)

    conv10, naming_factor = create_cbcb(conv9, f, init_mode, naming_factor)
    up4 = create_upsamping(conv10, naming_factor)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = 'conv2d_end',
                     kernel_initializer = init_mode)(up4) 
    # define autoencoder model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, f
if __name__ == '__main__':
    
    # Training
    autoencoder, f = create_custom_ae()
    
    # set checkpoints
    checkpoint_filepath = 'autoencoder_medium_weights_f_{}.h5'.format(f) 
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='loss',
        save_best_only=True)
    # fit the autoencoder model to reconstruct input
    epochs = 10
    history = autoencoder.fit(train_strips, batch_size=batch_size, epochs=epochs, callbacks = model_checkpoint_callback,
                              verbose=1)
    
    # Prediction
    # ae_pred, f  = load_model()
    ae_pred= load_model('autoencoder_medium_weights_f_{}.h5'.format(f) )
    ae_pred.compile(optimizer="adam", loss="mse", metrics = ['loss'])
    features = ae_pred.predict(train_strips.__getitem__(0)[0][0][np.newaxis])
    
    
    plt.figure()
    io.imshow(train_strips.__getitem__(0)[0][0])
    plt.xticks([])
    plt.yticks([])
    plt.title("Rectified Iris")
    plt.tight_layout()
    
    plt.figure()
    io.imshow(np.squeeze(features[0]))
    plt.xticks([])
    plt.yticks([])
    plt.title("U-Net Prediction")
    plt.tight_layout()
    
    # Model Save
    # autoencoder.save(
    #     'autoencoder.h5', overwrite=True, include_optimizer=True, save_format=None,
    #     signatures=None, options=None, save_traces=True)






