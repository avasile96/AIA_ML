# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:48:54 2021

@author: alex
"""

import os
from PIL import Image as PImage
import gc


class patient:
    def __init__(self, index):
        self.index = index


def loadImages(path):
    # return array of images from a directory (specified by "path")
    imagesList = os.listdir(path)
    loadedImages = []
    
    for image in imagesList:
        img = PImage.open(os.path.join(path, image))
        loadedImages.append(img)
        
    return loadedImages


def import_data(dataset_dir):
    # import ground truth and images (images are incorporated into the "patient" object)
    gt_dir = os.path.join(dataset_dir, 'groundtruth')
    gt = loadImages(gt_dir)
    
    patients_dir = os.path.join(dataset_dir, 'images')
    patient_indexes = [ name for name in os.listdir(patients_dir) if os.path.isdir(os.path.join(patients_dir, name)) ]
    
    patients = []
    for patient_index in patient_indexes:
        images_dir = os.path.join(patients_dir, patient_index)
        images = loadImages(images_dir)
        p = patient(patient_index)
        p.images = images
        patients.append(p)
    
    return patients, gt


if __name__ == '__main__':
    
    source_dir = os.path.dirname(os.path.abspath(__name__))
    project_dir = os.path.dirname(source_dir)
    dataset_dir = os.path.join(project_dir, 'dataset')

    patients, gt = import_data(dataset_dir)
    
    
    gc.collect()

    
