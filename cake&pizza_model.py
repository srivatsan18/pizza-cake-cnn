#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[3]:


data="C:\\Users\\Srivatsan\\Desktop\\pizza&cake"
cat=["cake","pizza"]


# In[5]:


train_cake_dir=os.path.join('C:\\Users\\Srivatsan\\Desktop\\pizza&cake\\cake')
train_pizza_dir=os.path.join('C:\\Users\\Srivatsan\\Desktop\\pizza&cake\\pizza')


# In[8]:


train_cake_names=os.listdir(train_cake_dir)
print(train_cake_names[:10])
train_pizza_names=os.listdir(train_pizza_dir)
print(train_pizza_names[:10])


# In[9]:


import matplotlib.image as mpimg


# In[10]:


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')     
])


# In[11]:


model.summary()


# In[12]:


from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy',
             optimizer=RMSprop(lr=0.001),
             metrics=['acc'])


# In[15]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255,
                                  rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'C:\\Users\\Srivatsan\\Desktop\\pizza&cake',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300*300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')


# In[16]:


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('acc')>0.99):
                    print("Reached 99% training so canceling!")
                    self.model.stop_training=True
callbacks=myCallback()


# In[19]:


history=model.fit_generator(train_generator,
                           steps_per_epoch=2,
                           epochs=150,verbose=1,
                           callbacks=[callbacks])

model.save('cake&pizza-CNN.model')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




