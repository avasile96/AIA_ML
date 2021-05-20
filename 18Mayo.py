import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image 
from tensorflow.keras.preprocessing.image import load_img
import keras
from skimage.color import gray2rgb
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
import random as rng

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

strip_img_paths = [
        os.path.join(dataset_dir, 'Strip', fname)
        for fname in os.listdir(os.path.join(dataset_dir, 'Strip'))
        if fname.endswith(".bmp") and not fname.startswith(".")]

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


def create_patients(dataset_dir):
    # create list of patients containing images from dataset
    
    gt_dir = os.path.join(dataset_dir, 'groundtruth')
    st_dir=os.path.join(dataset_dir,'Strip')
    patients_dir = os.path.join(dataset_dir, 'images')
    patients = []
    for patient_index in os.listdir(patients_dir):
        if os.path.isdir(os.path.join(patients_dir, patient_index)):
            gt = []
            st=[]
            p = patient(patient_index)
            p.images = loadImages(os.path.join(patients_dir, patient_index))
            for name in os.listdir(gt_dir):
                if patient_index in name:
                    gt.append(io.imread(os.path.join(gt_dir, name), as_gray = False))
            for name in os.listdir(st_dir):
                if patient_index in name:
                    st.append(io.imread(os.path.join(st_dir, name), as_gray = False))
                
            p.ground_truth = gt
            p.strip=st
            patients.append(p)
    return patients



def im_data_extract(list_of_patients):
    # Extracting images from patients
    x_train = []
    y_train = []
    for patient in list_of_patients:
        x_train.append(patient.strip)
        y_train.append(patient.index)
        
    #x_train_arr = np.array([gray2rgb(image) for sublist in x_train for image in sublist])
    #y_train_arr = np.array([gray2rgb(image) for sublist in y_train for image in sublist])
    x_train_arr=np.array([gray2rgb(image) for sublist in x_train for image in sublist])
    y_train_arr=np.array([int(sublist) for sublist in y_train for _ in range(1,11)])
    return x_train_arr, y_train_arr


#%% MAIN
if __name__ == '__main__':
    
    patients = create_patients(dataset_dir)
    
        #% Preparing training set
    from skimage.transform import rescale, resize, downscale_local_mean
    from skimage.color import gray2rgb
    
    x_arr, y_arr = im_data_extract(patients)
    
    #strip_image = patients[39].strip[0]
    #img = Image.fromarray(strip_image, 'RGB')
    #patient_number=int(patients[12].index)
    
    