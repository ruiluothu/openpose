import os
import glob
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from people_feature_generation import Person
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from sklearn.model_selection import KFold


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 30, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]      #20, 8, 3
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy


train_path = 'training_lstm'
actions = ['walking','running','other']
labels = np.array([0,1,2])
data, trainX, testX = np.zeros((0,20,8)), np.zeros((0,20,8)), np.zeros((0,20,8))
y, trainy, testy = np.zeros(0), np.zeros((0,3)), np.zeros((0,3))

for i in range(len(actions)):
    data = np.concatenate((data, np.load('%s.npy' % actions[i])), axis=0)
    y = np.concatenate((y, np.ones(data.shape[0])*labels[i]), axis=0)
y = to_categorical(y)

times = 10
accuracy = np.zeros(0)
# kf = KFold(n_splits=times)
# kf.get_n_splits(data)
# for train, test in kf.split(data):
total = data.shape[0]
a = np.arange(total)
for i in range(times):
    np.random.shuffle(a)
    train = a[0:int(round(total/10*9))]
    test = a[int(round(total/10*9)):-1]
    trainX = np.concatenate((trainX, data[train,:,:]), axis=0)
    testX = np.concatenate((testX, data[test,:,:]), axis=0)
    trainy = np.concatenate((trainy, y[train]), axis=0)
    testy = np.concatenate((testy, y[test]), axis=0)
    accuracy = np.concatenate((accuracy, np.array([evaluate_model(trainX, trainy, testX, testy)])))
print(accuracy)

