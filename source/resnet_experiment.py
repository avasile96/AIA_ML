# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:25:00 2021

@author: alex
"""

import tensorflow as tf


    
    resnet_model = tf.keras.applications.ResNet50(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000)