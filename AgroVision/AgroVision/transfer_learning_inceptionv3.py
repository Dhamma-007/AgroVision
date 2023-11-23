#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img,ImageDataGenerator
from tensorflow.keras.models import Sequential
from glob import glob


# In[2]:


# Load Dataset
train_set = "M:\\python\\plant-Disease-prediction-web-app-main\\plant-Disease-prediction-web-app-main\\dataset\\Cotton Disease\\train\\"
test_set = "M:\\python\\plant-Disease-prediction-web-app-main\\plant-Disease-prediction-web-app-main\\dataset\\Cotton Disease\\test\\"


# In[3]:


# Import the Inception V3 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights
IMAGE_SIZE = [224,224]
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[4]:


# Do not train existing weights

for layer in inception.layers:
    layer.trainable = False


# In[5]:


folders = glob("M:\\python\\plant-Disease-prediction-web-app-main\\plant-Disease-prediction-web-app-main\\dataset\\Cotton Disease\\train\\*") # len(folders) = 4


# In[6]:


# our layers - you can add more if you want
x = Flatten()(inception.output) #shape=(None, 51200)


# In[7]:


prediction = Dense(len(folders),activation='softmax')(x)


# In[8]:


# create a model object
model = Model(inputs=inception.input, outputs=prediction)


# In[9]:


# view the structure of the model
model.summary()


# In[10]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[11]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[12]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('M:\\python\\plant-Disease-prediction-web-app-main\\plant-Disease-prediction-web-app-main\\dataset\\Cotton Disease\\train\\',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('M:\\python\\plant-Disease-prediction-web-app-main\\plant-Disease-prediction-web-app-main\\dataset\\Cotton Disease\\test\\',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[13]:


incep = model.fit_generator(training_set,
                            validation_data = test_set,
                            epochs = 3,
                            steps_per_epoch = len(training_set),
                            validation_steps=len(test_set))


# In[14]:


## Plotting the Loss and Accuracy


# In[15]:


acc = incep.history['accuracy']
val_acc = incep.history['val_accuracy']
loss = incep.history['loss']
val_loss = incep.history['val_loss']
epochs=range(len(acc))


# In[16]:


plt.plot(epochs,acc,label='Train_acc',color='blue')
plt.plot(epochs,val_acc,label='Validation_acc',color='red')
plt.legend()
plt.title("Training and Validation Accuracy")


# In[17]:


plt.plot(epochs,loss,label='Train_loss',color='blue')
plt.plot(epochs,val_loss,label='Validation_loss',color='red')
plt.legend()
plt.title("Training and Validation loss")


# ## Prediction

# In[18]:


class_dict = {0:'diseased cotton leaf',
              1:'diseased cotton plant',
              2:'fresh cotton leaf',
              3:'fresh cotton plant' }


# In[19]:


import cv2
img_path =  'M:\\python\\plant-Disease-prediction-web-app-main\\plant-Disease-prediction-web-app-main\\dataset\\Cotton Disease\\test\\diseased cotton leaf\\dis_leaf (153)_iaip.jpg'
img = image.load_img(img_path, target_size=(224, 224))
 # Preprocessing the image
x = image.img_to_array(img)
# x = np.true_divide(x, 255)
## Scaling
x=x/255
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
preds=np.argmax(preds, axis=1)
pred_class = class_dict[preds[0]]

print(pred_class)


# In[20]:


# Save the model

model.save("incep.h5")


# In[ ]:




