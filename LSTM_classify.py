
# WIP
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import copy

model = keras.Sequential([keras.layers.Flatten(input_shape=tuple([1,10])),keras.layers.LSTM(20)])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def trainLSTM(vectorfile, resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    global model
    model = keras.Sequential([keras.layers.Flatten(input_shape=tuple([1,vlist.shape[1]])),keras.layers.LSTMCell(20)])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    parsed = []
    for v in vlist:
        parsed.append(np.array(v))
    model.fit(parsed, rlist, epochs=5)

def testLSTM(vectorfile,resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
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

if __name__ == '__main__':
    trainLSTM('la_pf_10.vvec','la_pf_10.rvec')
    print(testLSTM('la_pf_10.vvec','la_pf_10.rvec'))
    save('lstmm.mdl')