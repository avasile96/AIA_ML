# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 09:21:39 2021

@author: alex

UNET TRAINING
"""


import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import keras
from skimage.color import gray2rgb
from skimage.transform import resize


tf.debugging.set_log_device_placement(True)


img_size = (240, 320)
num_classes = 2
batch_size = 16

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

        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path, as_gray = True)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = img
            y[j] = tf.math.divide(y[j],255)
        return x, y

def ale_suffle(liist):
    train = []
    val = []
    total = []
    for i in range(0,2250,10):
        np.random.seed(42+i)
        a = liist[i:i+10]
        np.random.shuffle(a)
        total.extend(a)
        train.extend(a[0:5])
        val.extend(a[5:10])
    
    return train, val

def get_model(img_size, num_classes):
    #Build the model
    IMG_HEIGHT = img_size[0]
    IMG_WIDTH = img_size[1]
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



if __name__ == '__main__':
    
    # Split our img paths into a training and a validation set
    train_input_img_paths, val_input_img_paths = ale_suffle(input_img_paths)
    train_target_img_paths, val_target_img_paths = ale_suffle(target_img_paths)
    
    # Instantiate data Sequences for each split
    train_gen = IrisImageDatabase(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    val_gen = IrisImageDatabase(batch_size, img_size, val_input_img_paths, val_target_img_paths)

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()
    
    # Build model - UNet
    model = get_model(img_size, num_classes)
    model.summary()
    
    MTRX =[tf.keras.metrics.AUC(curve = 'ROC', name = "auc"),
           "accuracy",
           tf.keras.metrics.TruePositives(name = "TP"),
           tf.keras.metrics.TrueNegatives(name = "TN"),
           tf.keras.metrics.FalsePositives(name = "FP"),
           tf.keras.metrics.FalseNegatives(name = "FN"),
           tf.keras.metrics.Precision(name = "prec"),
           tf.keras.metrics.Recall(name = "rec")
           ]
        
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics = MTRX)
    
    import datetime
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Modelcheckpoint
    checkpointer = tf.keras.callbacks.ModelCheckpoint('iris_unet.h5', verbose=1, save_best_only=True)

    callbacks = [
        checkpointer,
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
        ]
    
    # Train the model, doing validation at the end of each epoch.
    epochs = 200
    hist = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks, batch_size = batch_size)
    # model.save(
    # 'iris_unet.h5', overwrite=True, include_optimizer=True, save_format=None,
    # signatures=None, options=None, save_traces=True)
    
    print (hist.history)
    
    keras.backend.clear_session()
    
    # preds_train = model.predict(train_gen)
    gc.collect()

    #%% Plotting
    import matplotlib.pyplot as plt

    # Loss
    y_ax = np.linspace(0,100,len(hist.history["accuracy"]), dtype = np.int)
    x_ax = np.linspace(0,len(hist.history["accuracy"]),len(hist.history["accuracy"]), dtype = np.int)
    
    plt.figure()
    lss, = plt.plot(x_ax,np.array(hist.history["loss"]), label='Training Loss')
    val_lss, = plt.plot(x_ax,np.array(hist.history["val_loss"]), label='Validation Loss')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('epochs')
    plt.title('UNet Segmentation Loss')
    
    # Accuracy
    plt.figure()
    acc, = plt.plot(x_ax,np.array(hist.history["accuracy"])*100, label='Training Accuracy')
    val_acc, = plt.plot(x_ax,np.array(hist.history["val_accuracy"])*100, label='Validation Accuracy')
    plt.legend(handles=[acc, val_acc])
    plt.xlabel('epochs')
    plt.ylabel('[%]')
    plt.title('UNet Segmentation Accuracy')
    
    print("The best Training Accuracy was {}".format(max(hist.history["accuracy"])))
    print("The best Validation Accuracy was {}".format(max(hist.history["val_accuracy"])))
    
    print("The best Training Loss was {}".format(min(hist.history["loss"])))
    print("The best Validation Loss was {}".format(min(hist.history["val_loss"])))
    
    # print("The best Training Loss was {}".format(min(history.history["auc"])))
    # print("The best Validation Loss was {}".format(min(history.history["val_auc"])))
    
    # F1
    precision = np.array(hist.history["prec"], dtype = np.float)
    recal = np.array(hist.history["rec"], dtype = np.float)
    F1 = 2*(precision*recal/(precision+recal))
    precision_val = np.array(hist.history["val_prec"], dtype = np.float)
    recal_val = np.array(hist.history["val_rec"], dtype = np.float)
    F1_val = 2*(precision_val*recal_val/(precision_val+recal_val))
    
    print("The best Training F1 was {}".format(np.mean(F1[-5:-1])))
    print("The best Validation F1 was {}".format(np.mean(F1_val[-5:-1])))
    
    plt.figure()
    lss, = plt.plot(x_ax, F1, label='Training F1')
    val_lss, = plt.plot(x_ax, F1_val, label='Validation F1')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('epochs')
    plt.title('UNet Segmentation F1')
    
    # False Positive Rate FP/FP+TN
    false_pos =  np.array(hist.history["FP"], dtype = np.float)
    true_neg =  np.array(hist.history["TN"], dtype = np.float)
    FPR = false_pos/(false_pos+true_neg)
    
    false_pos_val =  np.array(hist.history["val_FP"], dtype = np.float)
    true_neg_val =  np.array(hist.history["val_TN"], dtype = np.float)
    FPR_val = false_pos_val/(false_pos_val+true_neg_val)
    
    plt.figure()
    lss, = plt.plot(x_ax, FPR, label='Training FPR')
    val_lss, = plt.plot(x_ax, FPR_val, label='Validation FPR')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('epochs')
    plt.title('UNet Segmentation FPR')
    
    # True Positive Rate TP/TP+FN
    true_pos =  np.array(hist.history["TP"], dtype = np.float)
    false_neg =  np.array(hist.history["FN"], dtype = np.float)
    TPR = true_pos/(true_pos+false_neg)
    
    ture_pos_val =  np.array(hist.history["val_TP"], dtype = np.float)
    false_neg_val =  np.array(hist.history["val_FN"], dtype = np.float)
    TPR_val = ture_pos_val/(ture_pos_val+false_neg_val)
    
    plt.figure()
    lss, = plt.plot(x_ax, TPR, label='Training FPR')
    val_lss, = plt.plot(x_ax, TPR_val, label='Validation FPR')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('epochs')
    plt.title('UNet Segmentation TPR')
    
    # ROC (TPR vs TPR)
    plt.figure()
    lss, = plt.plot(TPR , FPR, label='Training ROC')
    val_lss, = plt.plot(TPR_val, FPR_val, label='Validation ROC')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('TPR')
    plt.ylabel('FPR')
    plt.title('UNet Segmentation ROC')
    
    # False Negative Rate FN/FN+TP
    FNR = false_neg/(true_pos+false_neg)
    FNR_val = false_neg_val/(ture_pos_val+false_neg_val)
    
    plt.figure()
    lss, = plt.plot(x_ax, FNR, label='Training FPR')
    val_lss, = plt.plot(x_ax, FNR_val, label='Validation FPR')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('epochs')
    plt.title('UNet Segmentation TPR')
    
    # ROC (TPR vs FNR)
    plt.figure()
    lss, = plt.plot(FNR , FPR, label='Training ROC')
    val_lss, = plt.plot(TPR_val, FPR_val, label='Validation ROC')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('FNR')
    plt.ylabel('FPR')
    plt.title('UNet Segmentation ROC (TPR vs FNR)')
    
    
    
    
    