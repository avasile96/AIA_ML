# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:20:23 2021

@author: vasil
"""

import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import keras
from skimage.color import gray2rgb
from skimage.transform import rescale, resize, downscale_local_mean
import cv2

tf.debugging.set_log_device_placement(True)


img_size = (240, 320)
num_classes = 256
batch_size = 10

source_dir = os.path.dirname(os.path.abspath(__name__))
project_dir = os.path.dirname(source_dir)
dataset_dir = os.path.join(project_dir, 'dataset')

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

def downsample(img_arr, desired_dimesion = [224, 224]):
    # Downsampling routine for stacks of images
    # img arr = array of images to which you want to apply the downsampling
    # desired_dimension = tuple of new dimension of the images
    img_arr_ds = np.array([resize(image, desired_dimesion) for image in img_arr])
    return img_arr_ds

def create_patients(dataset_dir):
    # create list of patients containing images from dataset
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

def im_data_extract(list_of_patients):
    # Extracting images from patients
    # list_of_patients = list of patient objeccts
    x_train = []
    y_train = []
    for patient in list_of_patients:
        x_train.append(patient.images)
        y_train.append(patient.ground_truth)
        
    x_train_arr = np.array([gray2rgb(image) for sublist in x_train for image in sublist])
    y_train_arr = np.array([gray2rgb(image) for sublist in y_train for image in sublist])
    
    return x_train_arr, y_train_arr

########################################################################################################

def getPolar2CartImg(image, rad):
    imgSize = image.shape
    c = (float(imgSize[0]/2.0), float(imgSize[1]/2.0))
    # imgRes = cv2.CreateImage((rad*3, int(360)), 8, 3)
    imgRes = np.zeros((rad*3, 360,3))
    #cv.LogPolar(image,imgRes,c,50.0, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS)
    cv2.warpPolar(image, imgRes, c, 60.0, cv2.WARP_POLAR_LOG)
    return (imgRes)

def getCircles(image):
	i = 80
	while i < 151:
		storage = cv2.CreateMat(image.width, 1, cv2.CV_32FC3)
		cv2.HoughCircles(image, storage, cv2.CV_HOUGH_GRADIENT, 2, 100.0, 30, i, 100, 140)
		circles = np.asarray(storage)
		if (len(circles) == 1):
			return circles
		i +=1
	return ([])

if __name__ == '__main__':
    
    patients = create_patients(dataset_dir)
    
    #% Preparing training set
    from skimage.transform import rescale, resize, downscale_local_mean
    from skimage.color import gray2rgb
    
    x_arr, y_arr = im_data_extract(patients)
    x_gray = cv2.cvtColor(x_arr[0], cv2.COLOR_BGR2GRAY)
    # cv2.imshow('og_image',x_arr[0])

    #%% Iris Strip
    

    strip = getPolar2CartImg(x_gray, 30)
    cv2.imshow('Contours', strip)
    
    
    
    



