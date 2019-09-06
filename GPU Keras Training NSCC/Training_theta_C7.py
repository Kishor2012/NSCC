#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 09:32:34 2019

@author: Kishor
"""

# Importing libraries
import numpy as np
import itertools
import pandas as pd
from numpy.random import seed
from numpy.random import rand
# importing libraries
import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils.vis_utils import plot_model
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
from keras import backend as K

#seed(1) # Seed for data reproducibility 
seed = 7
np.random.seed(seed)
data = pd.read_csv('thetadata_c7_1.txt') 
data = data.sample(frac=1)
print(data.shape)
dataset = data .values
# split into input (X) and output (Y) variables
X = dataset[:,0:7].astype(float)
Y = dataset[:,7]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# train-test split
train_data, test_data, train_targets, test_targets = train_test_split(X, encoded_Y, test_size=0.33)
# K-fold cross-validation
k = 2
num_val_samples = len(train_data) // k

#Building model
def build_model():
    model = models.Sequential()
    model.add(Dense(14, input_dim=7,activation='sigmoid'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))
    model.add(Dense(2, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# Some memory clean-up
K.clear_session()

num_epochs = 20000
val_acc_histories = []
val_loss_histories = []
tr_acc_histories = []
tr_loss_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1000, verbose=1)
    acc_history_v = history.history['val_acc']
    loss_history_v = history.history['val_loss']
    acc_history_t = history.history['acc']
    loss_history_t = history.history['loss']
    val_acc_histories.append(acc_history_v)
    val_loss_histories.append(loss_history_v)
    tr_acc_histories.append(acc_history_t)
    tr_loss_histories.append(loss_history_t)
    
    
average_acc_history_v = [np.mean([x[i] for x in val_acc_histories]) for i in range(num_epochs)]
average_loss_history_v = [np.mean([x[i] for x in val_loss_histories]) for i in range(num_epochs)]
average_acc_history_t = [np.mean([x[i] for x in tr_acc_histories]) for i in range(num_epochs)]
average_loss_history_t = [np.mean([x[i] for x in tr_loss_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_acc_history_v) + 1), average_acc_history_v)
plt.plot(range(1, len(average_acc_history_t) + 1), average_acc_history_t)
#plt.plot(range(1, len(average_loss_history_v) + 1), average_loss_history_v)
plt.xlabel('Epochs')
#plt.ylabel('Validation accuracy')
#plt.ylabel('Validation loss')
plt.ylabel('Metric')
plt.show()

test_loss, test_acc = model.evaluate(test_data, test_targets)
print('The accuracy is',test_acc)
