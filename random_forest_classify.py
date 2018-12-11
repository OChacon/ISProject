import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

"A Simple Random forest Classifier that can be used and trained to identify subreddits and save and load models"

rmf = RandomForestClassifier(n_estimators=30,max_depth=2)

def trainRMF(vectorfile, resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    global rmf
    rmf = RandomForestClassifier(n_estimators=400,max_depth=4)
    rmf.fit(vlist,rlist)

def testRMF(vectorfile, resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    global rmf
    error = sum(np.abs(rmf.predict(vlist) - rlist))
    return error

def save(filename):
    global rmf
    with open(filename, 'wb+') as f:
        pickle.dump(rmf, f)

def load(filename):
    global rmf
    with open(filename, 'rb') as f:
        rmf = pickle.load(f)

if __name__ == '__main__':
    #trainRMF('trainData.vvec','trainData.rvec')
    load('rmf.mdl')
    print(testRMF('testData.vvec','testData.rvec'))
    save('rmf.mdl')