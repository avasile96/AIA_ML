# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:25:00 2021

@author: alex
"""

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