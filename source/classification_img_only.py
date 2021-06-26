# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:03:27 2021

@author: alex
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
    model.add(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(img_size[0],img_size[1],1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(MaxPooling2D((2, 2)))
    model.add(MaxPooling2D((2, 2)))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(225, activation='softmax', kernel_initializer='he_uniform'))
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
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', mode='min', min_delta=0.01) # mostly out of time considerations
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
    
    a = model.predict(x_val)
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
    
    # False Positive Rate FP/FP+TN
    false_pos =  np.array(history.history["FP"], dtype = np.float)
    true_neg =  np.array(history.history["TN"], dtype = np.float)
    FPR = false_pos/(false_pos+true_neg)
    
    false_pos_val =  np.array(history.history["val_FP"], dtype = np.float)
    true_neg_val =  np.array(history.history["val_TN"], dtype = np.float)
    FPR_val = false_pos_val/(false_pos_val+true_neg_val)
    
    plt.figure()
    lss, = plt.plot(x_ax, FPR, label='Training FPR')
    val_lss, = plt.plot(x_ax, FPR_val, label='Validation FPR')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('epochs')
    plt.title('Shallow Net Classification FPR')
    
    # True Positive Rate TP/TP+FN
    true_pos =  np.array(history.history["TP"], dtype = np.float)
    false_neg =  np.array(history.history["FN"], dtype = np.float)
    TPR = true_pos/(true_pos+false_neg)
    
    ture_pos_val =  np.array(history.history["val_TP"], dtype = np.float)
    false_neg_val =  np.array(history.history["val_FN"], dtype = np.float)
    TPR_val = ture_pos_val/(ture_pos_val+false_neg_val)
    
    plt.figure()
    lss, = plt.plot(x_ax, TPR, label='Training FPR')
    val_lss, = plt.plot(x_ax, TPR_val, label='Validation FPR')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('epochs')
    plt.title('Shallow Net Classification TPR')
    
    # ROC (TPR vs TPR)
    plt.figure()
    lss, = plt.plot(TPR , FPR, label='Training ROC')
    val_lss, = plt.plot(TPR_val, FPR_val, label='Validation ROC')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('TPR')
    plt.ylabel('FPR')
    plt.title('Shallow Net Classification ROC')
    
    # False Negative Rate FN/FN+TP
    FNR = false_neg/(true_pos+false_neg)
    FNR_val = false_neg_val/(ture_pos_val+false_neg_val)
    
    plt.figure()
    lss, = plt.plot(x_ax, FNR, label='Training FPR')
    val_lss, = plt.plot(x_ax, FNR_val, label='Validation FPR')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('epochs')
    plt.title('Shallow Net Classification FPR')
    
    # ROC (FPR vs FNR)
    plt.figure()
    lss, = plt.plot(FNR , FPR, label='Training ROC')
    val_lss, = plt.plot(TPR_val, FPR_val, label='Validation ROC')
    plt.legend(handles=[lss, val_lss])
    plt.xlabel('FNR')
    plt.ylabel('FPR')
    plt.title('Shallow Net Classification ROC (TPR vs FNR)')
    
    