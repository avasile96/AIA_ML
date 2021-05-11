# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:41:36 2021

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