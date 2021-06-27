# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:03:27 2021

@author: alex

This script performs recognition (classification) using strip images directly.
"""

import os
import gc
import numpy as np
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from unet_manual import create_patients
from unet_manual_exp import ale_suffle
from classification_ae_features import DI_calculation
from sklearn.metrics import roc_curve, auc, det_curve, multilabel_confusion_matrix

tf.debugging.set_log_device_placement(True)


img_size = (240, 320)
num_classes = 224
batch_size = 64

source_dir = os.path.dirname(os.path.abspath(__name__))
project_dir = os.path.dirname(source_dir)
big_file_dir = os.path.dirname(project_dir)
strip_folder = os.path.join(big_file_dir, 'strips')
db_path = project_dir+"\\deep_features.csv"


def get_strips(strip_folder):
    strips = []
    pat_indices = []
    for strip_file in os.listdir(strip_folder):
        pat_indices.append(strip_file[8:11])
        strips.append(io.imread(os.path.join(strip_folder, strip_file)))
        
    return strips, pat_indices


def define_model():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(img_size[0],img_size[1],1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(225, activation='softmax'))
    return model

if __name__ == '__main__':
    
    #%% Data Reading
    strips, indices = get_strips(strip_folder)
    
    strips = np.array(strips, dtype = np.float)
    frame = np.zeros((2240,240,320,1))
    frame[:,:,:,0] = strips

    train_x = frame
    train_y = to_categorical(np.array(indices), 225)
    
    del strips, frame, indices
    gc.collect()
    
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
    model = define_model()
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
    
    callbacks = [
        # checkpointer,
        # tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss') # mostly out of time considerations
        # tf.keras.callbacks.TensorBoard(log_dir='logs')
        ]
    
    model.compile(loss=loss_fn, optimizer='adam', metrics=MTRX)

    n_ep = 200
    # Train 
    history = model.fit(x=x_train, y=y_train, 
                        validation_data=(x_val,y_val), 
                        epochs=n_ep, verbose = 2, 
                        batch_size = batch_size, 
                        callbacks = callbacks,
                        shuffle = True)
    
    y_val_pred = model.predict(x_val)
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
    fpr_det = dict()
    fnr_det = dict()
    roc_auc = dict()
    
    # Getting the fpr and tpr
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val[i, :], y_val_pred[i, :])
        fpr_det[i], fnr_det[i], _ = det_curve(y_val[i, :], y_val_pred[i, :])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(y_val.ravel(), y_val_pred.ravel())
    fpr_det["micro"], fnr_det["micro"], _ = det_curve(y_val.ravel(), y_val_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    all_fpr_det = np.unique(np.concatenate([fpr_det[i] for i in range(num_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_fnr_det = np.zeros_like(all_fpr_det)
    for i in range(num_classes):
        mean_fnr_det += np.interp(all_fpr_det, fpr_det[i], fnr_det[i])
    
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    mean_fnr_det /= num_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    fpr_det["macro"] = all_fpr_det
    fnr_det["macro"] = mean_fnr_det
    
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),linewidth=2)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]), linewidth=2)
    
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
    vss, = plt.plot(fpr["micro"], label='FPR')
    lss, = plt.plot(tpr["micro"], label='TPR')
    plt.legend(handles=[vss, lss])
    plt.xlabel('dunno')
    plt.title('Shallow Net Classification')
    
    # Plot all DET curves
    plt.figure()
    lss, = plt.plot(fpr_det["micro"], fnr_det["micro"],
             label='micro-average DET curve')

    plt.legend(handles=[lss])
    
    # Calculating EER
    diff_vect = (fnr_det['micro'] - fpr_det['micro'])**2
    min_diff_idx = np.where(diff_vect == np.min(diff_vect))
    EER = fnr_det['micro'][min_diff_idx]
    print("The EER was {}".format(EER))
    
    # Calculating DI
    y_val_pred_th = np.array(y_val_pred>0.5, dtype = np.int)
    DI_rf = DI_calculation(y_val, y_val_pred_th, y_val_pred)
    print("The DI was {}".format(DI_rf))
    