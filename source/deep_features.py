# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:36:53 2021

@author: vasil
"""

import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
import keras
import random
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
from unet_import import GetInputGenerator, unet_seg, polar_transform

tf.debugging.set_log_device_placement(True)


img_size = (240, 320)
num_classes = 2
batch_size = 1

source_dir = os.path.dirname(os.path.abspath(__name__))
project_dir = os.path.dirname(source_dir)
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
    
    # Generating UNet input
    unet_input, labels = GetInputGenerator(input_img_paths, batch_size, img_size, return_labels = True)
    
    #%% SEGMENTATION
    fluffy_seg = unet_seg(unet_input, batch_size)
    strips = []
    
    im_from_gen = unet_input.__getitem__(42)[0] # getting og image
    img_from_seg = fluffy_seg[42] #getting seg image
    
    f = plt.figure()
    f.suptitle("im_from_gen")
    io.imshow(im_from_gen)
    
    f = plt.figure()
    f.suptitle("img_from_seg")
    io.imshow(img_from_seg)
    
    strips.append(polar_transform(im_from_gen, img_from_seg))
    
    f = plt.figure()
    f.suptitle("strips")
    io.imshow(strips[0])
        
    #%% FEATURE EXTRACTION
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.preprocessing.image import load_img
    
    x = np.squeeze(fluffy_seg)*np.squeeze(unet_input) # masks * og_imgs
    
    print("[INFO] loading network...")
    res_model = ResNet50(weights="imagenet", include_top=False)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    res_model.fit(x = x, y = labels)
    
    features = res_model.predict(img_from_seg)
    features = features.reshape((features.shape[0], 7 * 7 * 2048))

