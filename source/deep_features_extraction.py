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
import cv2
import matplotlib.pyplot as plt
from unet_import import GetInputGenerator, unet_seg, polar_transform
from skimage.color import gray2rgb

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
        
    #%% PREPROCESSING
    from tensorflow.keras.applications import ResNet50

    dim = (224, 224)
    
    x_resized = np.zeros([2240,224,224,3])

    x = gray2rgb(np.squeeze(fluffy_seg)*np.squeeze(unet_input)) # masks * og_imgs
    
    for idx in range(x_resized.shape[0]): 
        x_resized[idx, :, :] = cv2.resize(x[idx], dsize = dim, interpolation = cv2.INTER_AREA)


    #%% FEATURE EXTRACTION
    
    del unet_input, fluffy_seg, x, im_from_gen, img_from_seg
    
    print("[INFO] loading network...")
    res_model = ResNet50(weights="imagenet", include_top=False)
    
    features = res_model.predict(x_resized, batch_size=batch_size)
    features_mod = features.reshape((features.shape[0], 7 * 7 * 2048))
    
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
    

