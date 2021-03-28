import tensorflow as tf
from tensorflow import keras
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from numpy import argmax


x=np.zeros((183,96,96,3),dtype=np.float32)
ye=np.zeros((183,1),dtype=np.float32)

p=0
for filename in os.listdir('files/players'):
    for player in os.listdir('files/players'+'/'+filename):
        img=cv2.imread('files/players'+'/'+filename+'/'+player)
        x[p]=cv2.resize(img,(96,96))
        ye[p]=int(filename)
        p=p+1
y = to_categorical(ye)
print(y)


x=x/255.0

from sklearn.utils import shuffle

v, u = shuffle(x, y)
inputs=keras.layers.Input((96,96,3))
l1=keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu')(inputs)
l2=keras.layers.MaxPool2D((2,2))(l1)


l3=keras.layers.Conv2D(16,(3,3),activation='relu')(l2)
l4=keras.layers.MaxPool2D((2,2))(l3)

l5=keras.layers.Conv2D(8,(3,3),activation='relu')(l4)
l6=keras.layers.MaxPool2D((2,2))(l5)

l7=keras.layers.Conv2D(2,(3,3),activation='relu')(l6)


l8=keras.layers.Flatten()(l7)
l9=keras.layers.Dense(256,activation='relu')(l8)
l10=keras.layers.Dropout(0.5)(l9)
l11=keras.layers.Dense(3,activation='sigmoid')(l10)
model=keras.Model(inputs=inputs,outputs=l11)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(v,u,6,500)

o=110
yu=model.predict(np.reshape(v[o],(1,96,96,3)))
plt.imshow(v[o])
print(argmax(yu))
print(argmax(u[o]))

model.save('model4.h5')
