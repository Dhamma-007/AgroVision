
# In[1]:


import tensorflow as tf


# In[2]:




# In[3]:


#Checking the version of tensorflow
tf.__version__


# In[4]:


#Importing the required libraries

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img

import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[5]:


from google.colab import drive 
drive.mount('/content/drive')


# In[6]:


import os
os.chdir('drive/MyDrive/plant_data')   #Changing the path directory to our dataset folder location


# In[7]:


get_ipython().system('pwd   #Checking the path directory')


# In[8]:


#Re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = '/content/drive/My Drive/plant_data/train'   #Path of train dataset
valid_path = '/content/drive/My Drive/plant_data/test'    #Path of test dataset


# In[9]:


#Import the ResNet50 library as shown below and add preprocessing layer to the front
#Here we will be using imagenet weights

resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[10]:


#Don't train the existing weights as it is a pre-trained model and we will be training our last layer alone
for layer in resnet.layers:
    layer.trainable = False


# In[11]:


#Getting the number of output classes
folders = glob('/content/drive/My Drive/plant_data/train/*')


# In[12]:


#We are flattening it so that we can add any number of layers in the last node
x = Flatten()(resnet.output)


# In[13]:


#Checking the length of folders
len(folders)


# In[14]:


#Predicting the data using length of folders and activation function used is softmax as there are more than 2 categories
prediction = Dense(len(folders), activation='softmax')(x)

#Creating a model object
model = Model(inputs=resnet.input, outputs=prediction)


# In[15]:


#Checking the structure of the model
model.summary()


# In[16]:


#Compiling the model with loss, optimitizer and metrics value
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[17]:


#Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Train data
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

#Test data
test_datagen = ImageDataGenerator(rescale = 1./255)


# In[18]:


#Make sure you provide the same target size as initialied for the image size
#Reading the number of images in train dataset
training_set = train_datagen.flow_from_directory('/content/drive/My Drive/plant_data/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[19]:


#Reading the number of images in test dataset
test_set = test_datagen.flow_from_directory('/content/drive/My Drive/plant_data/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[20]:


#Fit the model
r = model.fit(training_set, validation_data=test_set, epochs=20, steps_per_epoch=len(training_set), validation_steps=len(test_set))


# In[21]:


#Plotting the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

#Plotting the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[22]:


#Saving it as a h5 file
from tensorflow.keras.models import load_model
model.save('model_resnet55.h5')


# In[23]:


#Prediction using test set
y_pred = model.predict(test_set)


# In[24]:


y_pred







