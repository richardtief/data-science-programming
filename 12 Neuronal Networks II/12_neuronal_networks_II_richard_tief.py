#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import random

from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from tensorflow.keras import utils
from tensorflow.keras import losses


# In[3]:


(train_x, train_y), (test_x, test_y) = mnist.load_data()


# In[4]:


def reshape_x(x):
    shape = x.shape
    # Just one color channel
    return x.reshape(*shape, 1)

def reshape_y(y):
    # categorical classes for softmax, one-hot encoding
    # The identity array is a square array with ones on the main diagonal
    return np.array(list(map(lambda x: np.identity(10)[x], y)))

train_x_r = reshape_x(train_x)
train_y_r = reshape_y(train_y)

test_x_r = reshape_x(test_x)
test_y_r = reshape_y(test_y)

input_shape = (*train_x.shape[-2:], 1)
print(input_shape)
# (28, 28, 1) # Keras brauch das als Input, also Plus Tiefe


# In[14]:


plt.figure(figsize=(20,6))
for i, img in enumerate(train_x):
    plt.subplot(3,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_x[i])
    if i == 29:
        break
plt.show()


# In[6]:


BATCH = 32
EPOCHS = 12

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(), # multiple dimensions into one
    # layers.Dense(128, activation='relu' --> to cover all flattened layers
    layers.Dense(10, activation='softmax') # fully-connected, distribution
])
model.compile(loss=losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy']) # Human-readable metric
model.fit(train_x_r, train_y_r, batch_size=BATCH, epochs=EPOCHS)


# In[7]:


model.save('mnist_model.h5')
model = models.load_model('mnist_model.h5')
model.summary()


# In[9]:

# model.evaluate
prediction = model.predict(test_x_r)

plt.figure(figsize=(20,24))
for i, img in enumerate(test_x):
    c = max(enumerate(prediction[i]), key=lambda x: x[1])

    plt.subplot(9,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_x[i])
    plt.xlabel("class: " + str(c[0]) + "\npred: " + str(c[1]))
    if i == 89:
        break
plt.show()
