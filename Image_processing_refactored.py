#!/usr/bin/env python
# coding: utf-8

# In[101]:


import tensorflow as tf
import pandas as pd
from pandas import DataFrame
import numpy as np
import PIL as pl
from PIL import Image
from pathlib import Path
import csv


# In[ ]:





# In[ ]:





# In[102]:


def convert_to_matrix(images, label, image_data, labels):
    for img in images:
        # -- open the image -- resize the opened image -- convert the resized image to array
        img_arr  = np.array(Image.open(img).resize((64, 64), Image.ANTIALIAS))
        # --- before appending to list, check if each of element of image array are of same shape (64,64,3)
        if img_arr.shape == (64,64,3):
            image_data.append(img_arr)
            labels.append(label)
        
    return image_data, labels

def get_images(name):
    P = Path()
    path = P / 'data' /'train'/name
    name_images = list(path.glob('**/*.jpg'))
    print(type(name_images))
    
    return name_images

def write_to_csv(image_list):
    with open("imageData2.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(image_list)
    


# In[103]:


image_data   = []
labels = []

imagesa = get_images('alien')
image_data, labels = convert_to_matrix(imagesa, 0, image_data, labels)


imagesp = get_images('predator')
image_data,labels2 = convert_to_matrix(imagesp, 1, image_data, labels)

#final_images = image_data + image_data2
#final_labels = labels + labels2
final_labels = np.array(labels).reshape(693,1)

write_to_csv(np.append(np.array(image_data).reshape(-1,12288), final_labels, axis =1))
data = pd.read_csv('imageData2.csv')
labels = data['0']
data.drop(columns = '0', axis = 1, inplace = True)
data = np.array(data)
labels = np.array(labels)

features = data.reshape(-1,64,64,3)
labels = labels.reshape(692,1)


# In[104]:


features.shape


# In[105]:


labels.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




