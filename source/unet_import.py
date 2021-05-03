# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:56:55 2021

@author: alex

This script imports the Unet and applies segmentation

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
import random
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
from func_lib import get_circles, mean_shift
from im_proc import draw_circles
from scipy.spatial import distance

tf.debugging.set_log_device_placement(True)


img_size = (240, 320)
num_classes = 2
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

class IrisImageDatabase(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        # x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path, as_gray = True)
            x[j] = img
        # y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        y = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = img
            y[j] = tf.math.divide(y[j],255)
        return x, y
    
def apply_segmentation_gen(generator): # TODO
    img_roi = 2 ### PLACEHOLDER
    return img_roi

def GetTestTrainGenerators(val_percent, input_img_paths, target_img_paths, batch_size, img_size):
    val_samples = int(len(target_img_paths)*val_percent/100)
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]
    # Instantiate data Sequences for each split
    train_gen = IrisImageDatabase(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    val_gen = IrisImageDatabase(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    return train_gen, val_gen

def stripTease(seg_img, center, max_radius): # TODO
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    final_strip = cv2.linearPolar(seg_img, center, max_radius, flags)
    
    return final_strip

def get_model(img_size, num_classes):
    #Build the model
    IMG_HEIGHT = 240
    IMG_WIDTH = 320
    IMG_CHANNELS = 1
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    
    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
     
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
     
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

def find_contours(og_image, thresh):
    contours,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours
    
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)

        contour_list.append(contour)
    
    cv2.drawContours(og_image, contour_list,  -1, (255,0,0), 2)
    cv2.imshow('Objects Detected',og_image)
    
    return contour_list

if __name__ == '__main__':
    
    # Train-Validation Split
    train_gen, val_gen = GetTestTrainGenerators(1, input_img_paths, target_img_paths, batch_size, img_size)

    #%% SEGMENTATION
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()
    # Import U-Net
    
    model = load_model('iris_unet.h5')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics = ['accuracy'])
    # Get predictions (segment) images from the dataset
    # "a" contains images segmented by UNet
    a = model.predict(
        val_gen, 
        batch_size=2, 
        verbose=2, 
        steps=None, 
        callbacks=None, 
        max_queue_size=10,
        workers=2, 
        use_multiprocessing=False)
    #%% Strip
    """
    val_gen.__getitem__(0)[0][0] DESCRIPTION
    ...
    First index represents the input/target pair selection (0 = first pair)
    Second index represents the input/target selection (0 is input)
    Thrid index is the n-th image in the minibatch (0 = first image)
    --> so in the line below we access the first image of the first
    input of the first input/target pair 
    """
    og_image = val_gen.__getitem__(0)[0][0] # getting og image
    prediction = a[0] # getting prediction

    f1 = plt.figure()
    f1.suptitle('og_image')
    io.imshow(og_image)
    
    # tresholding
    th1, binary_pred = cv2.threshold(np.squeeze(prediction),0.1,1,cv2.THRESH_BINARY)
    binary_pred = np.uint8(binary_pred)
    f2 = plt.figure()
    f2.suptitle('binary_pred')
    io.imshow(binary_pred)
    
    # opening
    kern_radius = 5
    kernel = np.ones((kern_radius,kern_radius),np.uint8)
    open_mask = cv2.morphologyEx(binary_pred, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    # multiplying threshold with og image
    cut_image = np.multiply(og_image, open_mask)
    f3 = plt.figure()
    f3.suptitle('cut_image')
    io.imshow(cut_image)        
    
    # loading random image from database for tests
    tst_img = io.imread(input_img_paths[0])
    tst_gray = cv2.cvtColor(tst_img, cv2.COLOR_RGB2GRAY)
    
    # # contour trial
    # cnt = find_contours(tst_img, open_mask)
    # cv2.drawContours(tst_img, cnt,  -1, (255,0,0), 2)
    # cv2.imshow('Objects Detected',tst_img)
    
    #%% Hough Circles Trial
    pred_sq = np.squeeze(prediction)*255
    pred_sq_uint8 = np.uint8(pred_sq)
    pred_sq_uint8_mShift = mean_shift(pred_sq_uint8)
    # pred_sq_uint8_mShift_RGB = cv2.cvtColor(pred_sq_uint8_mShift, cv2.COLOR_GRAY2BGR)

    
    og_copy = cv2.cvtColor(og_image, cv2.COLOR_GRAY2BGR)
    f2 = plt.figure()
    f2.suptitle('pred_sq_uint8')
    io.imshow(pred_sq_uint8)
    
    # # loading random image from database for tests
    # tst_img = io.imread(input_img_paths[0])
    # tst_gray = cv2.cvtColor(tst_img, cv2.COLOR_RGB2GRAY)
    
    # Hough Circles
    pupil_outline = cv2.HoughCircles(pred_sq_uint8, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=20, maxRadius=50)
    pupil_outline = np.uint16(np.around(pupil_outline))
    
    
    iris_outline = iris_outline = cv2.HoughCircles(pred_sq_uint8, cv2.HOUGH_GRADIENT, 1, 2, minRadius = 300)
    iris_outline = np.uint16(np.around(pupil_outline))
    
    canvas = np.ones_like(og_image)

    for i in pupil_outline[0, :]:
            # draw the outer circle
            cv2.circle(og_copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(og_copy, (i[0], i[1]), 2, (0, 0, 255), 3)    
    for i in iris_outline[0, :]:
            # draw the outer circle
            cv2.circle(og_copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(og_copy, (i[0], i[1]), 2, (0, 0, 255), 3) 
    f3 = plt.figure()
    f3.suptitle('circle_img')
    io.imshow(og_copy)
    

    
    center = (np.squeeze(pupil_outline)[0], np.squeeze(pupil_outline)[1])
    max_rad = np.squeeze(iris_outline)[2]
    
    strip = stripTease(og_image, center, max_rad*2)
    f4 = plt.figure()
    f4.suptitle('strip')
    io.imshow(strip)
    #%%
    # # distance map
    # inverted_mask = cv2.bitwise_not(open_mask)-254
    # dist_im = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 3)
    # cv2.normalize(dist_im, dist_im, 0, 1.0, cv2.NORM_MINMAX)
    # f4 = plt.figure()
    # f4.suptitle('distance map')
    # io.imshow(dist_im)
    
    # closest_distance = 9999
    # center_of_image = (np.floor(img_size[0]/2), np.floor(img_size[1]/2))
    # euc = distance.euclidean((0,0), center_of_image)
    # brightest_point = dist_im[0,0]
    # imVal_sort = np.sort(dist_im)
    
    # all_dist = []
    
    # for i in range(img_size[0]):
    #     for j in range(img_size[1]):
    #         all_dist.append(distance.euclidean((i,j), center_of_image))
    
    # all_dist_array = np.reshape(all_dist, img_size)
    # for i in range(img_size[0]):
    #     for j in range(img_size[1]):
    #         if 
            
    # for i in range(img_size[0]):
    #     for j in range(img_size[1]):
    #         if (dist_im[i,j]>dist_im[i-1,j-1]):
    #             if (distance.euclidean((i,j), center_of_image) < closest_distance):
    #                 closest_distance = distance.euclidean((i,j), center_of_image)
    #                 pupil_center = (i,j)
    
    # f4 = plt.figure()
    # f4.suptitle('strip')
    # io.imshow(strip)
    
    #%% POLAR TRANSFORM



