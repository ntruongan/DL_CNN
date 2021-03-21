# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 15:26:40 2021

@author: ntruo
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

# Part 1 - Data Preprocessing

#%% Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('trainingSet.tar/trainingSet',
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 
                                                 class_mode = 'categorical')

# #%% Preprocessing the Test set
# test_datagen = ImageDataGenerator(rescale = 1./255)
# test_set = test_datagen.flow_from_directory('dataset/test_set',
#                                             target_size = (150, 150),
#                                             batch_size = 32,
#                                             class_mode = 'categorical')

#%%

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[150, 150, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Part 3 - Training the CNN
# ca
# Compiling the CNN
cnn.compile(optimizer = 'adam', 
            loss = 'categorical_crossentropy', 
            metrics = ['accuracy'])

#%%
# Training the CNN on the Training set and evaluating it on the Test set
his = cnn.fit(x = training_set, epochs = 25)