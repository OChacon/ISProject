
# WIP
import tensorflow as tf
import tensorflow
import scipy
from tensorflow import keras
import pickle
import numpy as np
import copy
#tf.nn.rnn_cell.BasicLSTMCell(64)
#tf.nn.static_rnn()


#model = keras.Sequential([keras.layers.Flatten(input_shape=tuple([1,10])),keras.layers.LSTM(20)])
#model.compile(optimizer=tf.train.AdamOptimizer(),
#              loss='sparse_categorical_crossentropy',
#             metrics=['accuracy'])
model = None

class batchdata:
    def __init__(self, ilist, rlist, bs):
        self.data = ilist
        self.labels = rlist.tolist()
        hotlbls = np.zeros((len(self.labels),2))
        hotlbls[np.arange(len(self.labels)), self.labels] = 1
        self.labels = hotlbls
        self.batch_count = 0
        self.batch_size = bs

    def next(self):
        if self.batch_count == len(self.data):
            self.batch_count = 0
        bdata = self.data[self.batch_count:min(len(self.data),self.batch_count+self.batch_size)]
        blable = self.labels[self.batch_count:min(len(self.data),self.batch_count+self.batch_size)]
        self.batch_count = min(len(self.data),self.batch_count+self.batch_size)
        return bdata, blable


class LSTMmodel:
    def __init__(self, d):
        self.learning_rate = 0.01
        self.training_steps = 10000
        self.batch_size = 5
        self.display_step = 200

        # Network Parameters
        self.seq_max_len = d.data[0].shape[0]  # Sequence max length
        self.n_hidden = 64  # hidden layer num of features
        self.n_classes = 2  # linear sequence or not
        self.data = d

        self.x = tf.placeholder("float", [None, self.seq_max_len, 1])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # Define weights
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        self.xs = tf.unstack(self.x, self.seq_max_len, 1)
        self.cell = tf.nn.rnn_cell.LSTMCell(64, name='basic_lstm_cell')
        self.out, self.state = tf.nn.static_rnn(self.cell, self.xs, dtype=tf.float32)
        self.out = tf.stack(self.out)
        self.out = tf.transpose(self.out, [1, 0, 2])
        self.index = tf.range(0, self.batch_size) * self.seq_max_len + (self.seq_max_len - 1)
        self.out = tf.gather(tf.reshape(self.out, [-1, self.n_hidden]), self.index)
        self.pred = tensorflow.matmul(self.out,  self.weights['out']) + self.biases['out']

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

    def train(self):
        with tf.Session() as sess:
            # Run the initializer
            sess.run(self.init)

            for step in range(1, self.training_steps + 1):
                batch_x, batch_y = self.data.next()
                # Run optimization op (backprop)
                sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
                if step % self.display_step == 0 or step == 1:
                    # Calculate batch accuracy & loss
                    acc, loss = sess.run([self.accuracy, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})
                    print("Step " + str(step * self.batch_size) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(acc))

            print("Optimization Finished!")

    def test(self,dataset):
        with tf.Session() as sess:
            sess.run(self.init)
            return sess.run(self.accuracy, feed_dict={self.x: dataset.data, self.y: dataset.labels})

def trainLSTM(vectorfile, resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    act = []
    for x in vlist:
        act.append(x.todense().reshape(x.shape[1],x.shape[0]))
    traindata = batchdata(act,rlist, 5)



    global model
    model = LSTMmodel(traindata)
    model.train()

def testLSTM(vectorfile,resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    act = []
    for x in vlist:
        act.append(x.todense().reshape(x.shape[1], x.shape[0]))
    global model
    testdata = batchdata(act,rlist, 5)
    acc = model.test(testdata)
    return acc

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