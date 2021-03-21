# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:18:22 2021

@author: ntruo
"""

import pandas as pd
import cv2
import numpy as np


df = pd.read_csv("all_letters_info.csv")
#%%
path = r'all_letters_image/all_letters_image/'

#%%
y = df['label'].values
X = np.zeros((y.shape[0],32,32,3))
X = []

for i, file in enumerate(df['file'].values): 
    img = cv2.imread(path+file)
    img = cv2.resize(img, (32, 32))
    X.append(img)
#%%

cv2.imshow('img', X[123])

#%%

from keras.utils import to_categorical
X = np.array(X)
y = np.array(to_categorical(y))
#%%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

#%%

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

train_datagen.fit(X_train)


training_set = train_datagen.flow(X_train, y_train, batch_size=32)
#%%
test_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen.fit(X_test)
test_set = test_datagen.flow(X_test, y_test, batch_size=32)

#%%




#%% Part 2 - Building the CNN
import tensorflow as tf
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[32, 32, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
cnn.add(tf.keras.layers.Dropout(rate = 0.3))
# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=y.shape[1], activation='softmax'))

# Part 3 - Training the CNN
# ca
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ['accuracy'])
#%%
# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(training_set, validation_data = test_set, epochs = 50)

# Part 4 - Making a single prediction

#%%

import cv2
from numpy import  argmax
img = cv2.imread("11_79.png")
x_pred = []
img = cv2.resize(img, (32, 32))
x_pred.append(img)
x_pred = np.array(x_pred)
y_pred = cnn.predict(x_pred)
print(argmax(y_pred))