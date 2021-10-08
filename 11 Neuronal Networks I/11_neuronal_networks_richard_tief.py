#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models, layers
from tensorflow.keras import utils


# In[14]:


# using predefined train/test split 
(train_X, train_y), (test_X, test_y) = boston_housing.load_data()


# In[15]:


def build_model(width, depth):
    model = models.Sequential()
    model.add(layers.Input(shape=(13,), name='Input'))
    for layer in range(depth):
        units = width / (2**layer)
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1, name='Output'))

    return model


# In[16]:


model = build_model(512, 2)
model.compile(optimizer='adam', loss='mse')


# In[17]:


model.summary() 


# In[18]:


BATCH = 32
EPOCHS = 100


hist1 = model.fit(train_X, train_y, batch_size=BATCH, epochs=EPOCHS, shuffle=True, verbose=0) # shuffle=True
err = model.evaluate(test_X, test_y)


# In[19]:


X_min = np.concatenate((train_X, test_X)).min(0)
X_max = np.concatenate((train_X, test_X)).max(0)
train_X_norm = (train_X - X_min) / (X_max - X_min)
test_X_norm = (test_X - X_min) / (X_max - X_min)


# In[20]:


hist2 = model.fit(train_X_norm, train_y, batch_size=BATCH, epochs=EPOCHS, verbose=0)
err_norm = model.evaluate(test_X_norm, test_y)


# In[21]:


print('Error:           ', err)
print('Normalized error:', err_norm)

# Error:            26.51251792907715
# Normalized error: 20.916975021362305


# In[22]:


# Grid search to find the best combination

losses = dict()

for width, depth in product((128, 256, 512, 1024), (1, 2, 3, 4, 5)):
    model = build_model(width, depth)
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_X_norm, train_y, batch_size=BATCH, epochs=EPOCHS, verbose=0)
    
    loss = model.evaluate(test_X_norm, test_y)
    losses[width, depth] = loss
    
    
# 4/4 [==============================] - 0s 940us/step - loss: 27.8296
# 4/4 [==============================] - 0s 976us/step - loss: 24.6232
# 4/4 [==============================] - 0s 893us/step - loss: 23.0448
# 4/4 [==============================] - 0s 1ms/step - loss: 20.4022
# 4/4 [==============================] - 0s 899us/step - loss: 23.4917
# 4/4 [==============================] - 0s 896us/step - loss: 24.1293
# 4/4 [==============================] - 0s 830us/step - loss: 23.2223
# 4/4 [==============================] - 0s 878us/step - loss: 18.3992
# 4/4 [==============================] - 0s 748us/step - loss: 19.5727
# 4/4 [==============================] - 0s 1ms/step - loss: 16.5293
# 4/4 [==============================] - 0s 752us/step - loss: 23.8600
# 4/4 [==============================] - 0s 951us/step - loss: 21.0448
# 4/4 [==============================] - 0s 984us/step - loss: 21.6048
# 4/4 [==============================] - 0s 1ms/step - loss: 14.8044
# 4/4 [==============================] - 0s 1ms/step - loss: 17.0576
# 4/4 [==============================] - 0s 929us/step - loss: 25.3742
# 4/4 [==============================] - 0s 1ms/step - loss: 19.5648
# 4/4 [==============================] - 0s 1ms/step - loss: 14.8753
# 4/4 [==============================] - 0s 2ms/step - loss: 14.3364
# 4/4 [==============================] - 0s 1ms/step - loss: 13.8320


# In[23]:


# sort by best combination
sort_dict= dict(sorted((value, key) for (key,value) in losses.items())) 
for k,v in sort_dict.items():
    print(v, "     ", k)
    
# (1024, 5)      13.831974983215332
# (1024, 4)      14.336359024047852
# (512, 4)       14.804420471191406
# (1024, 3)      14.875267028808594
# (256, 5)       16.5292911529541
# (512, 5)       17.057615280151367
# (256, 3)       18.39919090270996
# (1024, 2)      19.56477165222168
# (256, 4)       19.572736740112305
# (128, 4)       20.40224838256836
# (512, 2)       21.044843673706055
# (512, 3)       21.604833602905273
# (128, 3)       23.04477310180664
# (256, 2)       23.222307205200195
# (128, 5)       23.491689682006836
# (512, 1)       23.860002517700195
# (256, 1)       24.129302978515625
# (128, 2)       24.62320899963379
# (1024, 1)      25.37423324584961
# (128, 1)       27.829570770263672

