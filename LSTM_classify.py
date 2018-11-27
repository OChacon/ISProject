import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np

model = keras.Sequential([keras.layers.LSTM(20)])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def trainLSTM(vectorfile, resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    global model
    model = keras.Sequential([keras.layers.LSTM(20)])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(vlist, rlist, epochs=5)

def testLSTM(vectorfile,resultsfile):
    vlist = pickle.load(vectorfile)
    rlist = pickle.load(resultsfile)
    loss, acc = model.evaluate(vlist, rlist)
    return loss

def save(filename):
    global model
    with open(filename, 'wb+') as f:
        pickle.dump(model, f)

def load(filename):
    global model
    with open(filename, 'rb') as f:
        model = pickle.load(f)
