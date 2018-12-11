"""
A not so simple LSTM implimentation with heavy influence from the internet and lots of trial and error.
Heavily based on the work found here
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/dynamic_rnn.ipynb
It uses tensor flow and can train and save and load a model to help identify subreddits or really any classifiers.
It requires small 200 length feature vectors to run in reasonable time.

Author: Mike Hurlbutt

"""


import tensorflow as tf
import tensorflow
import pickle
import numpy as np
#tf.nn.rnn_cell.BasicLSTMCell(64)
#tf.nn.static_rnn()




model = None
indices = []

"""
Batchifies a simple two lists of data so tensor flow doesn't complain.
"""
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


"""
The model file and all the logic involved
"""
class LSTMmodel:
    """
    A tone of setup with d being the batchified data
    """
    def __init__(self, d):
        self.learning_rate = 0.01
        self.training_steps = 15000
        self.batch_size = 25
        self.display_step = 200
        self.loaded = False

        # Network Parameters
        self.seq_max_len = d.data[0].shape[0]  # Sequence max length
        self.n_hidden = 16   # hidden layer num of features
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

        # The actual lstm implimentation wildly inefficent
        self.xs = tf.unstack(self.x, self.seq_max_len, 1)
        self.cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, name='basic_lstm_cell')
        self.out, self.state = tf.nn.static_rnn(self.cell, self.xs, dtype=tf.float32)
        self.out = tf.stack(self.out)
        self.out = tf.transpose(self.out, [1, 0, 2])
        batch_size = tf.shape(self.out)[0]
        self.index = tf.range(0, batch_size) * self.seq_max_len + (self.seq_max_len - 1)
        self.out = tf.gather(tf.reshape(self.out, [-1, self.n_hidden]), self.index)
        #self.out = tf.reshape(self.out, [-1, self.n_hidden])
        self.pred = tensorflow.matmul(self.out,  self.weights['out']) + self.biases['out']

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # Initialize the variables and set up savor and session
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    """
    Trains the actual model with a set of multiple batches
    """
    def train(self):
        #if self.sess is None:
        #    self.sess = tf.Session()

        # Run the initializer
        if not self.loaded:
            self.sess.run(self.init)

        for step in range(1, self.training_steps + 1):
            batch_x, batch_y = self.data.next()
            # Run optimization op (backprop)
            self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
            if step % self.display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                acc, loss = self.sess.run([self.accuracy, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})
                print("Step " + str(step * self.batch_size) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

    """
    Test a batchified dataset and return the accuracy
    """
    def test(self,dataset):
        #with tf.Session() as sess:
        #    sess.run(self.init)
        return self.sess.run(self.accuracy, feed_dict={self.x: dataset.data, self.y: dataset.labels})

    def close(self):
        self.sess.close()

    def save(self,filename):
        self.saver.save(self.sess,filename)

    def load(self,filename):
        self.loaded = True
        self.saver.restore(self.sess,filename)


"""
Set up data and train the model
"""
def trainLSTM(vectorfile, resultsfile, modelfile=None):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    act = []
    #samples = np.random.random_sample(x.shape[0])
    for x in vlist:
        act.append(x.todense().reshape(x.shape[1],x.shape[0]))#[:min(x.shape[1],100)])
    traindata = batchdata(act,rlist, 25)



    global model
    model = LSTMmodel(traindata)
    if modelfile is not None:
        load(modelfile)
    model.train()


"""
Set up data and feed into the model
"""
def testLSTM(vectorfile,resultsfile,loadfile = ""):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    act = []
    for x in vlist:
        act.append(x.todense().reshape(x.shape[1], x.shape[0]))#[:min(x.shape[1], 100)])
    global model
    testdata = batchdata(act,rlist, 50)
    if model is None:
        model = LSTMmodel(testdata)
        load(loadfile)
    acc = model.test(testdata)
    return np.rint(50 - acc*50)


"""
Save and load the tensorflow session
"""
def save(filename):
    global model
    model.save(filename)

def load(filename):
    global model

    model.load(filename)

if __name__ == '__main__':
    trainLSTM('trainData200.vvec','trainData200.rvec','./lstm.mdl')
    i = testLSTM('testData200.vvec','testData200.rvec')
    print(i)
    if i < 10:
        save('./lstm.mdl')