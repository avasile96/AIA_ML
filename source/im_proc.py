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
import random as rng

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

def getPolar2CartImg(image, rad):
	imgSize = cv2.GetSize(image)
	c = (float(imgSize[0]/2.0), float(imgSize[1]/2.0))
	imgRes = cv2.CreateImage((rad*3, int(360)), 8, 3)
	cv2.LogPolar(image,imgRes,c,60.0, cv2.CV_INTER_LINEAR+cv2.CV_WARP_FILL_OUTLIERS)
	return (imgRes)


def contrast_adaptive(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1

def k_means_seg(img, k = 3):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))    
    return res2

def get_contours(img_in):
    # Detect edges using Canny
    th = 100
    canny_output = cv2.Canny(img_in, th, th * 2)
    cv2.imshow('canny_output',canny_output)
    
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    color = (0,0,0)
    cv2.drawContours(x_gray, contours, 3, color, -2, cv2.LINE_8, hierarchy, 0)
    return contours

def mean_shift(ms_in):
    ms_in = cv2.cvtColor(ms_in, cv2.COLOR_GRAY2BGR)
    ms_img = cv2.pyrMeanShiftFiltering(ms_in, 25, 30)
    ms_img = cv2.cvtColor(ms_img, cv2.COLOR_BGR2GRAY)
    return ms_img

def morfo_trans(image):
    # Grayscale morphology
    kern_radius = 5
    kernel = np.ones((kern_radius,kern_radius),np.uint8)
    
    closing = cv2.morphologyEx(x_gray, cv2.MORPH_CLOSE, kernel, iterations = 2)
    # cv2.imshow('closing',closing)
    cl_op = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations = 2)
    # cv2.imshow('opening',cl_op)
    return cl_op

def draw_circles(img):

    medianBlurim = cv2.medianBlur(img, 5)
    pupil_outline = cv2.HoughCircles(medianBlurim, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=20, maxRadius=120)

    pupil_outline = np.uint16(np.around(pupil_outline))
    
    # Mean shift filtering
    m_shift = mean_shift(img)
    
    iris_outline = cv2.HoughCircles(m_shift, cv2.HOUGH_GRADIENT, 1, 400, param1=100, param2=50)
    print(iris_outline)
    iris_outline = np.uint16(np.around(iris_outline))
    
    for i in iris_outline[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    for i in pupil_outline[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    return img, iris_outline, pupil_outline

def iris_strip(img, pupil_outline, iris_outline):
    center = pupil_outline[0,0,0],pupil_outline[0,0,1]
    iris_radius = iris_outline[0,0,2]-pupil_outline[0,0,2] # outter_radius - inner radius
    
    canvas = np.zeros_like(img)
    cv2.circle(canvas,center,iris_radius,255)
    points = np.transpose(np.where(img==255))
    
    nsamples = points.shape[0]
    samples = np.linspace(0, 2 * np.pi, nsamples)[:-1]
    
    polar = np.zeros((iris_radius, nsamples))

    for r in range(iris_radius):
        for (i, theta) in (range(samples.shape[0]), samples):
            x = r * np.cos(theta) + points[i][0]
            y = r * np.sin(theta) + points[i][1]
            
            polar[r][theta * nsamples / 2.0 / np.pi] = img[y][x]
    
    # plt.figure(figsize=(10, 5))
    # plt.imshow(polar, cmap='gray')
    return polar


#%% MAIN
if __name__ == '__main__':
    
    patients = create_patients(dataset_dir)
    
    #% Preparing training set
    from skimage.transform import rescale, resize, downscale_local_mean
    from skimage.color import gray2rgb
    
    x_arr, y_arr = im_data_extract(patients)

    og_image = patients[94].images[0]
    # cv2.imshow('og_image',x_arr[0])
    
    x_gray = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img2gray',x_gray)

    #%% Morphological Transformations
    cl_op = morfo_trans(og_image)
    cv2.imshow('opening',cl_op)

    #%% Finding contours
    # contour_image = get_contours(x_gray)
    # cv2.imshow('Contours', contour_image)
    
    #%% CIRCLES
    eye_circles, pupil_outline, iris_outlline = draw_circles(x_gray)
    cv2.imshow('eye_circles', eye_circles)
    

    #%% MEAN SHIFT
    ms_img = mean_shift(cl_op)
    cv2.imshow('mean_shifted_image', ms_img)
    
    #%% KMEANS
    k_seg_out = k_means_seg(ms_img, k = 4)
    cv2.imshow('k_means_seg', k_seg_out)
    
    
    #%% Iris Strip
    strip = iris_strip(eye_circles, pupil_outline, iris_outlline)
    cv2.imshow('Contours', strip)
    
    
    
    
    



