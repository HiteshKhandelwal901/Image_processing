#!/usr/bin/env python
# coding: utf-8

# In[36]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import PIL as pl
from PIL import Image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[38]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense

model = tf.keras.Sequential()
model.add(Input((16,16,3)))
model.add(Conv2D(8, (8, 8), padding='same', activation='relu'))
model.add(MaxPooling2D((4,4)))
model.add(Conv2D(8, (8, 8), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(4, (4, 4), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='Adam' , loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:





# In[ ]:


# ------------------ EXAMPLE 1 -----------------

(training_features, training_labels), (test_features, test_labels) = cifar10.load_data()

num_classes = 10

training_labels = tf.keras.utils.to_categorical(training_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)


def get_generator(features, labels, batch_size=1):
    for n in range(int(len(features)/batch_size)):
        yield (features[n*batch_size:(n+1)*batch_size], labels[n*batch_size:(n+1)*batch_size])
        
def monochrome(x):
    def func_bw(a):
        average_colour = np.mean(a)
        return [average_colour, average_colour, average_colour]
    x = np.apply_along_axis(func_bw, -1, x)
    return x

datagen = ImageDataGenerator(preprocessing_function = monochrome,
                             rotation_range = 100,
                             rescale = 1/255.0)

datagen.fit(training_features)
dataiter = datagen.flow(training_features, training_labels, batch_size =1)
#model.fit(dataiter, .....)


# In[32]:



#--------------------- EXAMPLE-2 -------------------------------------
Train_path = 'PUT YOUR TRAIN DIRC PATH HERE'
val_path = 'PUT YOUR VAL DIRC PATH HERE'
datagenerator = ImageDataGenerator(rescale=(1/255.0))
train_generator = datagenerator.flow_from_directory(train_path, batch_size = 64, target_size = (16,16))
val_generator = datagenerator.flow_from_directory(val_path, batch_size = 64, target_size = (16,16))
train_steps_per_epoch = train_generator.n // train_generator.batch_size
val_steps = val_generator.n // val_generator.batch_size
model.fit_generator(train_generator, validation_data = val_generator, steps_per_epoch = train_steps_per_epoch)


# In[ ]:





# In[ ]:





# In[ ]:




