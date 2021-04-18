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


class patient:
    def __init__(self, index):
        self.index = index

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
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y

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

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    
    source_dir = os.path.dirname(os.path.abspath(__name__))
    project_dir = os.path.dirname(source_dir)
    dataset_dir = os.path.join(project_dir, 'dataset')

    patients = import_data(dataset_dir)
    
    # io.imshow(np.subtract(patients[0].images[0],
    #           np.multiply(patients[0].images[0], patients[0].ground_truth[0]))) # test that shows red iris
    
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
    
    # Split our img paths into a training and a validation set
    val_samples = np.floor(x_train_arr.shape[0] * 40/100)
    random.Random(1337).shuffle(x_train_arr)
    random.Random(1337).shuffle(y_train_arr)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]
    
    # Instantiate data Sequences for each split
    train_gen = OxfordPets(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    
    #%% RESNET (Machine Learning Algorithm)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    
    # # pipe_resnet = Pipeline([
    # #                         ("classifier_resnet", tf.keras.applications.ResNet50())
    # #                         ])
    
    # model_resnet = tf.keras.applications.ResNet50(include_top = False)
    # model_resnet.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    
    # # grid_params = [
    # #                 {"classifier": [tf.keras.applications.ResNet50()],
    # #                   "classifier__include_top": [True],
    # #                   "classifier__weights":['imagenet'],
    # #                   "classifier__input_tensor":[1e-3],
    # #                   "classifier__input_shape": [-1],
    # #                   "classifier__pooling":[None],
    # #                   "classifier__classes":[1000],
    # #                  }
    # #                 ]
    
    # model_resnet.fit(x_train_arr_ds[0:32], y = y_train_arr_ds[0:32])

    
    # gc.collect()
    
    #%% U-Net
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    model = get_model(img_size, num_classes)
    model.summary()
    
    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    
    callbacks = [
        keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
    ]
    
    # Train the model, doing validation at the end of each epoch.
    epochs = 15
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

