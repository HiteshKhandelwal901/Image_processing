#!/usr/bin/env python
# coding: utf-8

# In[150]:


import tensorflow as tf
import pandas as pd
from pandas import DataFrame
import numpy as np
import PIL as pl
from PIL import Image
from pathlib import Path
import csv


# In[151]:


def convert_to_matrix(images, label):
    image_data = []
    labels = []
    for i in range(len(images)):
        img      = images[i]
        imgOpen  = Image.open(img)
        resized_img = imgOpen.resize((64, 64), Image.ANTIALIAS)
        img_arr2 = np.array(resized_img)
        if img_arr2.shape == (64,64,3):
            #print("shape checked valid === ", img_arr2.shape)
            image_data.append(img_arr2)
            labels.append(label)
        
    return image_data, labels

def get_images(name):
    P = Path()
    path = P / 'data' /'train'/name
    name_images = list(path.glob('**/*.jpg'))
    print(type(name_images))
    
    return name_images

def write_to_csv(image_list):
    with open("imageData1.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(image_list)
    
    
    
    
    
    
    


# In[153]:


image_data   = []
image_data2  = []
final_images = []

imagesa = get_images('alien')
image_data, labels = convert_to_matrix(imagesa, 0)


imagesp = get_images('predator')
image_data2,labels2 = convert_to_matrix(imagesp, 1)

final_images = image_data + image_data2
final_labels = labels + labels2
final_labels = np.array(final_labels).reshape(693,1)

write_to_csv(np.append(np.array(final_images).reshape(-1,12288), final_labels, axis =1))
#data = pd.read_csv('imageData.csv')
#datanp = np.asarray(data)
#image_data = datanp.reshape(-1,64,64,3)
data = pd.read_csv('imageData1.csv')
labels = data['0']
data.drop(columns = '0', axis = 1, inplace = True)
data = np.array(data)
labels = np.array(labels)

features = data.reshape(-1,64,64,3)
labels = labels.reshape(692,1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




