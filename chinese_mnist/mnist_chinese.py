# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 14:07:07 2021

@author: ntruo
"""

import pandas as pd
import cv2 
# from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from  tensorflow.keras.utils import to_categorical
#%%
dataframe = pd.read_csv('chinese_mnist.csv')
# print(dataframe.columns)
index = dataframe.iloc[:,:-2].values 
value = dataframe["code"].values


#%%
# label_filename_link = {}
filename_label_link = {}

filename_list = []
label_list = []
X = []
y = [] 
for i in range(0,len(value)):
    x = index[i]
    filename = "input_%s_%s_%s.jpg" % (x[0], x[1], x[2])
    # print(filename)
    val = value[i] 
    # print(val)
    filename_label_link[filename] = val
    filename_list.append(filename)
    label_list.append(val)
    a = cv2.imread("data\data\%s"%filename)
    # a = imread
    X.append(a)
    y.append(val)
#%%
X = np.array(X)

y = np.array(y)
#%%



#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#%%
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test) 
#%%
from keras.models import Sequential
import tensorflow as tf

model = Sequential()
model.add(tf.keras.layers.Conv2D(128, 3,
                                 padding = "same", 
                                 activation = "relu", 
                                 input_shape=[64, 64, 3]))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2))

model.add(tf.keras.layers.Conv2D(128, 3,
                                 padding = "same", 
                                 activation = "relu", 
                                 input_shape=[64, 64, 3]))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides = 2))
model.add(tf.keras.layers.Flatten())

# model.add(tf.keras.layers.Dropout())
model.add(tf.keras.layers.Dense(128,activation = "relu"))
model.add(tf.keras.layers.Dense(16,activation = "softmax"))


                                                
#%%



# model.compile(optimizer='adam',loss = "categorical_crossentropy", metrics = ['accuracy'])  
model.compile(optimizer='adam',loss = "categorical_crossentropy", metrics = ['accuracy'])  


#%%



model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5)





#%%

a=cv2.imread("data\data\input_99_9_10.jpg")
#%%

















