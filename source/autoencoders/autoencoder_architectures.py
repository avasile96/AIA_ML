# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 01:42:21 2021

@author: vasil
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
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

img_size = (320,240)

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

