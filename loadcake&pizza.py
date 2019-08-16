#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

cat=["cake","pizza"]

def prepare(filepath):
    img_size=300
    img_array=cv2.imread(filepath,1)
    new_array=cv2.resize(img_array,(img_size,img_size))

    return new_array.reshape(-1,img_size,img_size,3)

model=tf.keras.models.load_model("cake&pizza-CNN.model")
img='C:\\Users\\Srivatsan\\Desktop\\pizza&cake\\test/101.jpg'
prediction=model.predict([prepare(img)])
print("\t\t\t\tTHE PREDICTION IS")
print(cat[int(prediction[0][0])])
res=cv2.imread(img)
plt.imshow(res)
plt.show()
#cv2.imshow("image",res)

