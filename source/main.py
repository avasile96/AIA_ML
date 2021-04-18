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

class patient:
    def __init__(self, index):
        self.index = index

def loadImages(path):
    # return array of images from a directory (specified by "path")
    imagesList = os.listdir(path)
    loadedImages = []
    for image in imagesList:
        img = io.imread(os.path.join(path, image), as_gray = True)
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
                    gt.append(io.imread(os.path.join(gt_dir, name), as_gray = True))
            p.ground_truth = gt
            patients.append(p)
    return patients


if __name__ == '__main__':
    
    source_dir = os.path.dirname(os.path.abspath(__name__))
    project_dir = os.path.dirname(source_dir)
    dataset_dir = os.path.join(project_dir, 'dataset')

    patients = import_data(dataset_dir)
    
    io.imshow(np.subtract(patients[0].images[0],
              np.multiply(patients[0].images[0], patients[0].ground_truth[0]))) # test that shows red iris
    
    #%% Preparing training set
    x_train = []
    y_train = []
    
    for patient in patients:
        x_train.append(patient.images)
        y_train.append(patient.ground_truth)
    
    x_train = [image for sublist in x_train for image in sublist]
    y_train = [image for sublist in y_train for image in sublist]
    
    #%% RESNET
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    
    # pipe_resnet = Pipeline([
    #                         ("classifier_resnet", tf.keras.applications.ResNet50())
    #                         ])
    
    model_resnet = tf.keras.applications.ResNet50()
    model_resnet.compile(optimizer='adam', loss='mse', metrics='accuracy')
    
    # grid_params = [
    #                 {"classifier": [tf.keras.applications.ResNet50()],
    #                   "classifier__include_top": [True],
    #                   "classifier__weights":['imagenet'],
    #                   "classifier__input_tensor":[1e-3],
    #                   "classifier__input_shape": [-1],
    #                   "classifier__pooling":[None],
    #                   "classifier__classes":[1000],
    #                  }
    #                 ]
    
    model_resnet.fit(x_train, y = y_train)

    
    gc.collect()
