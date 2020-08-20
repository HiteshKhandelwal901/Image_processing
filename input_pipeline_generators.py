#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
from pandas import DataFrame
import numpy as np
import PIL as pl
from PIL import Image
from pathlib import Path
import csv
import tensorflow as tf



def my_gen():
    for i in range(5000):
        features = np.random.randn(256)
        labels = np.random.randint(0,1, size = 1)
        yield features, labels 
        
def get_data(batch_size):
    for batch, (x,y) in enumerate(tf.data.Dataset.from_generator(my_gen, (tf.float32, tf.int16)).batch(batch_size)):
        print(y.shape)
        print(x.shape)
        yield x,y
        


# In[ ]:


model = Sequential()
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
model.fit_generator(get_data(500), steps_per_epoch = 50, epochs = 5)

