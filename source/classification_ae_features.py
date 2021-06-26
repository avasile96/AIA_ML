# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:41:36 2021

@author: alex

This script performs recognition (classification) using deep features.
"""

import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from unet_import import GetInputGenerator
from skimage.color import gray2rgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import keras
import csv
from unet_manual_exp import ale_suffle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle


tf.debugging.set_log_device_placement(True)


img_size = (240, 320)
num_classes = 224
batch_size = 64

source_dir = os.path.dirname(os.path.abspath(__name__))
project_dir = os.path.dirname(source_dir)
big_file_dir = os.path.dirname(project_dir)
dataset_dir = os.path.join(project_dir, 'dataset')
strip_folder = os.path.join(source_dir, 'strips')
db_path = project_dir+"\\deep_features.csv"


if __name__ == '__main__':
    
    #%% Data Reading
    train = pd.read_csv(db_path, index_col=(0))
    train_x = np.array(train)
    train_y = to_categorical(np.array(train.index), 225)
    
    #%% 50-50 Split
    x_train, x_val = ale_suffle(train_x)
    y_train, y_val = ale_suffle(train_y)
    
    # additional shuffling routine to break the task structure appart
    # we might run into the danger of the network learning the task structure
    np.random.seed(42)
    np.random.shuffle(x_train)
    
    np.random.seed(42)
    np.random.shuffle(y_train)
    
    np.random.seed(13)
    np.random.shuffle(x_val)
    
    np.random.seed(13)
    np.random.shuffle(y_val)
    
    x_train = np.array(x_train, dtype = np.float)
    x_val = np.array(x_val, dtype = np.float)
    
    y_train = np.array(y_train, dtype = np.int)
    y_val = np.array(y_val, dtype = np.int)
    
    plt.figure()
    io.imshow(y_train)
    plt.title('y_train')
    
    plt.figure()
    io.imshow(y_val)
    plt.title('y_val')
    
    del train_x, train_y
    gc.collect()
    
    
    #%% Shallow neural net
    """
    How did I come up with the values of 256 and 16 for the two hidden layers?
    A good rule of thumb is to take the square root of the previous number of nodes 
    in the layer and then find the closest power of 2.
    In this case, the closest power of 2 to 100352 is 256. The square root of 256
    is then 16, thus giving us our architecture definition.
    """
    # model = Sequential()
    # model.add(Dense(512, input_shape=(9600,), activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(Dense(225, activation="softmax"))
    # loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    model = Sequential()
    model.add(Dense(16, input_shape=(9600,), activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Dense(225, activation="softmax"))
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    MTRX =[tf.keras.metrics.AUC(curve = 'ROC', name = "auc"),
           tf.keras.metrics.CategoricalAccuracy(name='c_accuracy'),
           tf.keras.metrics.TruePositives(name = "TP"),
           tf.keras.metrics.TrueNegatives(name = "TN"),
           tf.keras.metrics.FalsePositives(name = "FP"),
           tf.keras.metrics.FalseNegatives(name = "FN"),
           tf.keras.metrics.Precision(name = "prec"),
           tf.keras.metrics.Recall(name = "rec")
           ]
    
    model.compile(loss=loss_fn, optimizer='adam', metrics=MTRX)
    n_ep = 300
    
    callbacks = [
    # checkpointer,
    # tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', mode='min', min_delta=0.1) # mostly out of time considerations
    # tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]
        
    # Train 
    history = model.fit(x=x_train, y=y_train, 
                        validation_data=(x_val,y_val), 
                        epochs=n_ep, verbose = 2, 
                        batch_size = batch_size, 
                        callbacks = callbacks,
                        shuffle = True)
    
    y_val_pred = model.predict(x_val)
    y_train_pred = model.predict(x_train)
    
    #%% Plotting & Metrics
    
    # Accuracy & Loss
    y_ax = np.linspace(0,100,len(history.history["c_accuracy"]), dtype = np.int)
    x_ax = np.linspace(0,len(history.history["c_accuracy"]),len(history.history["c_accuracy"]), dtype = np.int)
    
    plt.figure()
    lss, = plt.plot(x_ax,np.array(history.history["loss"]), label='Training Loss')
    val_lss, = plt.plot(x_ax,np.array(history.history["val_loss"]), label='Validation Loss')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('epochs')
    plt.title('Shallow Net Classification Loss')
    
    plt.figure()
    acc, = plt.plot(x_ax,np.array(history.history["c_accuracy"])*100, label='Training Accuracy')
    val_acc, = plt.plot(x_ax,np.array(history.history["val_c_accuracy"])*100, label='Validation Accuracy')
    plt.legend(handles=[acc, val_acc])
    plt.xlabel('epochs')
    plt.ylabel('[%]')
    plt.title('Shallow Net Classification Accuracy')
    
    print("The best Training Accuracy was {}".format(max(history.history["c_accuracy"])))
    print("The best Validation Accuracy was {}".format(max(history.history["val_c_accuracy"])))
    
    print("The best Training Loss was {}".format(min(history.history["loss"])))
    print("The best Validation Loss was {}".format(min(history.history["val_loss"])))
    
    # F1
    precision = np.array(history.history["prec"], dtype = np.float)
    recal = np.array(history.history["rec"], dtype = np.float)
    
    F1 = 2*(precision*recal/(precision+recal))
    precision_val = np.array(history.history["val_prec"], dtype = np.float)
    recal_val = np.array(history.history["val_rec"], dtype = np.float)
    F1_val = 2*(precision_val*recal_val/(precision_val+recal_val))
    
    plt.figure()
    lss, = plt.plot(x_ax, F1, label='Training F1')
    val_lss, = plt.plot(x_ax, F1_val, label='Validation F1')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('epochs')
    plt.title('Shallow Net Classification F1')
    
    # ROC (validation)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_train[i, :], y_train_pred[i, :])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(y_train.ravel(), y_train_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linewidth=2)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linewidth=2)
    lw = 2
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(num_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #              ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
    
    plt.figure()
    lss, = plt.plot(fpr["micro"], label='FPR')
    val_lss, = plt.plot(tpr["micro"], label='TPR')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('dunno')
    plt.title('Shallow Net Classification')
    
    plt.figure()
    lss, = plt.plot(fpr["micro"],tpr["micro"], label='dunno')
    plt.legend(handles=[lss])
    plt.xlabel('dunno')
    plt.title('Shallow Net Classification')
    
    
    