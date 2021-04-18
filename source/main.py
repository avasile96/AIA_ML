# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:48:54 2021

@author: alex
"""

import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import keras


class patient:
    def __init__(self, index):
        self.index = index

def loadImages(path):
    # return array of images from a directory (specified by "path")
    imagesList = os.listdir(path)
    loadedImages = []
    for image in imagesList:
        img = io.imread(os.path.join(path, image), as_gray = False)
        loadedImages.append(img)
    return loadedImages

def import_data(dataset_dir):
    # import ground truth and images (incorporated into the "patient" object)
    gt_dir = os.path.join(dataset_dir, 'groundtruth')
    patients_dir = os.path.join(dataset_dir, 'images')
    
    patients = []
    for patient_index in os.listdir(patients_dir):
        if os.path.isdir(os.path.join(patients_dir, patient_index)):
            gt = []
            p = patient(patient_index)
            p.images = loadImages(os.path.join(patients_dir, patient_index))
            for name in os.listdir(gt_dir):
                if patient_index in name:
                    gt.append(io.imread(os.path.join(gt_dir, name), as_gray = False))
            p.ground_truth = gt
            patients.append(p)
    return patients

if __name__ == '__main__':
    
    source_dir = os.path.dirname(os.path.abspath(__name__))
    project_dir = os.path.dirname(source_dir)
    dataset_dir = os.path.join(project_dir, 'dataset')

    patients = import_data(dataset_dir)
    
    
    #%% Preparing training set
    from skimage.transform import rescale, resize, downscale_local_mean
    from skimage.color import gray2rgb
    x_train = []
    y_train = []
    
    for patient in patients:
        x_train.append(patient.images)
        y_train.append(patient.ground_truth)
    
    x_train_arr = np.array([image for sublist in x_train for image in sublist])
    y_train_arr = np.array([gray2rgb(image) for sublist in y_train for image in sublist])
    
    # Downsampling routine (if needed)
    # x_train_arr_ds = []
    # y_train_arr_ds = []
    
    # x_train_arr_ds = np.array([resize(image, [224,224]) for image in x_train_arr])
    # y_train_arr_ds = np.array([resize(image, [224,224]) for image in y_train_arr])
    
    #%% VALIDATION SPLIT
    import random
    

    
    #%% MODEL TRAINING
    
    foo = (np.subtract(x_train_arr,
           np.multiply(x_train_arr, y_train_arr))) # test that shows red iris
    io.imshow(foo[0])
    
    #%% METRICS & PLOTWORK

