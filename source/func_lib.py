# -*- coding: utf-8 -*-
"""
Created on Sun May  2 12:31:17 2021

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
import random
import cv2

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

def daugman_normalizaiton(image, r_in, r_out, img_size):
    height = img_size[0]
    width = img_size[1]
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    r_out = r_in + r_out
    # Create empty flatten image
    flat = np.zeros((height,width, 3), np.uint8)
    circle_x = int(image.shape[0] / 2)
    circle_y = int(image.shape[1] / 2)

    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)

            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            color = image[int(Xc)][int(Yc)]  # color of the pixel

            flat[j][i] = color
    return flat  # liang

def mean_shift(ms_in):
    ms_in = cv2.cvtColor(ms_in, cv2.COLOR_GRAY2BGR)
    ms_img = cv2.pyrMeanShiftFiltering(ms_in, 25, 30)
    ms_img = cv2.cvtColor(ms_img, cv2.COLOR_BGR2GRAY)
    return ms_img

def get_circles(img):
    """
    Circle defined as: x_center, y_center, radius

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    img : TYPE
        DESCRIPTION.
    iris_outline : TYPE
        DESCRIPTION.
    pupil_outline : TYPE
        DESCRIPTION.

    """
    #% Hough Circles Trial
    pred_sq = np.squeeze(prediction)*255
    pred_sq_uint8 = np.uint8(pred_sq)
    pred_sq_uint8_mShift = mean_shift(pred_sq_uint8)
    # pred_sq_uint8_mShift_RGB = cv2.cvtColor(pred_sq_uint8_mShift, cv2.COLOR_GRAY2BGR)

        # HOUGH CIRCLES: getting pupil circle (x_center, y_center, rad)
    # pred_sq = np.squeeze(prediction)*255
    # pred_sq_uint8 = np.uint8(pred_sq)
    # open_mask_blur = cv2.GaussianBlur(open_mask,(5,5),0)
    # pupil_outline = cv2.HoughCircles(open_mask_blur, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=20, maxRadius=50)
    # if (pupil_outline is None):
    #     center = (np.floor(img_size[0]/2),np.floor(img_size[1]/2))
    # else:
    #     pupil_outline = np.uint16(np.around(pupil_outline))
    #     center = (np.squeeze(pupil_outline)[0], np.squeeze(pupil_outline)[1])
    
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
    
    
    iris_outline = cv2.HoughCircles(pred_sq_uint8, cv2.HOUGH_GRADIENT, 1, 2, minRadius = 180)
    iris_outline = np.uint16(np.around(iris_outline))
    
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

def stripTease(seg_img, center, max_radius):
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

def unet_seg():
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
    return fluffy_seg

def distance_map_trial():
    # distance map
    inverted_mask = cv2.bitwise_not(open_mask)-254
    dist_im = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 3)
    cv2.normalize(dist_im, dist_im, 0, 1.0, cv2.NORM_MINMAX)
    f4 = plt.figure()
    f4.suptitle('distance map')
    io.imshow(dist_im)
    
    closest_distance = 9999
    center_of_image = (np.floor(img_size[0]/2), np.floor(img_size[1]/2))
    euc = distance.euclidean((0,0), center_of_image)
    brightest_point = dist_im[0,0]
    imVal_sort = np.sort(dist_im)
    
    all_dist = []
    
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            all_dist.append(distance.euclidean((i,j), center_of_image))
    
    all_dist_array = np.reshape(all_dist, img_size)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            pass
            
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            if (dist_im[i,j]>dist_im[i-1,j-1]):
                if (distance.euclidean((i,j), center_of_image) < closest_distance):
                    closest_distance = distance.euclidean((i,j), center_of_image)
                    pupil_center = (i,j)
    
    f4 = plt.figure()
    f4.suptitle('strip')
    io.imshow(strip)
    return 

class PredictionData(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, return_labels):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.return_labels = return_labels

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        index = []
        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path, as_gray = True)
            x[j] = img
            index.append(batch_input_img_paths[j][26:29])
        # if (self.return_labels == True):
        #     return x, index
        # else:
            return x

def GetPredInput(input_img_paths, batch_size, img_size, return_labels = False):
    # random.Random(1337).shuffle(input_img_paths)
    input_img_paths = input_img_paths
    input_gen = PredictionData(batch_size, img_size, input_img_paths, return_labels = return_labels)
    return input_gen

#%% AUTOENCODER
def create_autoencoder(img_size, number_of_channels, init_mode):
    
    inChannel = number_of_channels
    input_img = Input(shape = (img_size[0], img_size[1], inChannel), name = 'input_3')
    f = 4
    
    conv1 = Conv2D(f, (3, 3), activation='relu', padding='same', name = 'conv1', 
                   kernel_initializer = init_mode)(input_img) #320 x 240 x f
    conv1 = BatchNormalization(name = 'batch_normalization_28')(conv1)
    conv1 = Conv2D(f*2, (3, 3), activation='relu', padding='same', name = 'conv2d_30',
                   kernel_initializer = init_mode)(conv1)
    conv1 = BatchNormalization(name = 'batch_normalization_29')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_4')(conv1) #160 x 120 x 2f
    conv2 = Conv2D(f*4, (3, 3), activation='relu', padding='same', name = 'conv2d_31',
                   kernel_initializer = init_mode)(pool1) #160 x 120 x 4f
    conv2 = BatchNormalization(name = 'batch_normalization_30')(conv2)
    conv2 = Conv2D(f*4, (3, 3), activation='relu', padding='same', name = 'conv2d_32',
                   kernel_initializer = init_mode)(conv2)
    conv2 = BatchNormalization(name = 'batch_normalization_31')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_5')(conv2) #80 x 60 x 4f
    conv3 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_33',
                   kernel_initializer = init_mode)(pool2) #80 x 60 x 8f (small and thick)
    conv3 = BatchNormalization(name = 'batch_normalization_32')(conv3)
    conv3 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_34',
                   kernel_initializer = init_mode)(conv3)
    conv3 = BatchNormalization(name = 'batch_normalization_33')(conv3)
    conv4 = Conv2D(f*16, (3, 3), activation='relu', padding='same', name = 'conv2d_35',
                   kernel_initializer = init_mode)(conv3) #80 x 60 x 16f (small and thick)
    conv4 = BatchNormalization(name = 'batch_normalization_34')(conv4)
    conv4 = Conv2D(f*16, (3, 3), activation='relu', padding='same', name = 'conv2d_36',
                   kernel_initializer = init_mode)(conv4)
    conv4 = BatchNormalization(name = 'batch_normalization_35')(conv4)
    pool3 = MaxPooling2D(pool_size=(2, 2), name = 'max_pooling2d_5')(conv2) #40 x 30 x 16f
    ########################################################
    conv5 = Conv2D(f*16, (3, 3), activation='relu', padding='same', name = 'conv2d_31',
                   kernel_initializer = init_mode)(pool3) #160 x 120 x 16f
    conv5 = BatchNormalization(name = 'batch_normalization_36')(conv2)
    conv5 = Conv2D(f*16, (3, 3), activation='relu', padding='same', name = 'conv2d_32',
                   kernel_initializer = init_mode)(conv2)
    conv5 = BatchNormalization(name = 'batch_normalization_37')(conv2)
    ########################################################
    conv5_res = Reshape((-1,1))(conv5) # Feature Layer
    conv5_shape = tuple(conv5.shape)    
    conv6 = Reshape((conv5_shape[1], conv5_shape[2], conv5_shape[3]))(conv5_res)
    ########################################################
    conv6 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_37',
                   kernel_initializer = init_mode)(conv6) #80 x 60 x 8f
    conv6 = BatchNormalization(name = 'batch_normalization_38')(conv6)
    conv6 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_38',
                   kernel_initializer = init_mode)(conv6)
    conv6 = BatchNormalization(name = 'batch_normalization_39')(conv6)
    #########################################################
    conv7 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_37',
                   kernel_initializer = init_mode)(conv5) #80 x 60 x 8f
    conv7 = BatchNormalization(name = 'batch_normalization_40')(conv7)
    conv7 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_38',
                   kernel_initializer = init_mode)(conv7)
    conv7 = BatchNormalization(name = 'batch_normalization_41')(conv7)
    conv8 = Conv2D(f*8, (3, 3), activation='relu', padding='same', name = 'conv2d_39',
                   kernel_initializer = init_mode)(conv7) #80 x 60 x 4f
    conv8 = BatchNormalization(name = 'batch_normalization_42')(conv8)
    conv8 = Conv2D(f*4, (3, 3), activation='relu', padding='same', name = 'conv2d_40',
                   kernel_initializer = init_mode)(conv8)
    conv8 = BatchNormalization(name = 'batch_normalization_43')(conv8)
    up1 = UpSampling2D((2,2), name = 'up_sampling2d_4')(conv8) #160 x 120 x 2f
    conv9 = Conv2D(f*2, (3, 3), activation='relu', padding='same', name = 'conv2d_41',
                   kernel_initializer = init_mode)(up1) # 160 x 120 x 2f
    conv9 = BatchNormalization(name = 'batch_normalization_44')(conv9)
    conv9 = Conv2D(f, (3, 3), activation='relu', padding='same', name = 'conv2d_42',
                   kernel_initializer = init_mode)(conv9)
    conv9 = BatchNormalization(name = 'batch_normalization_45')(conv9)
    up2 = UpSampling2D((2,2), name = 'up_sampling2d_5')(conv7) # 320 x 240 x f
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = 'conv2d_43',
                     kernel_initializer = init_mode)(up2) # 320 x 240 x 1
    # define autoencoder model
    autoencoder = Model(input_img, decoded)
    return autoencoder