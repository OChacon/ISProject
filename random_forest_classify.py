import pickle
from sklearn.ensemble import RandomForestClassifier

rmf = RandomForestClassifier(n_estimators=100,max_depth=4)

def trainRMF(vectorfile, resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    global rmf
    rmf = RandomForestClassifier(n_estimators=100,max_depth=4)
    rmf.fit(vlist,rlist)

def testSVM(vectorfile, resultsfile):
    vlist = pickle.load(vectorfile)
    rlist = pickle.load(resultsfile)
    error = 0
    global rmf
    for x in range(len(vlist)):
        v = vlist[x]
        r = rlist[x]
        p = rmf.predict(v)
        error += abs(p-r)
    return error

def save(filename):
    global rmf
    with open(filename, 'wb+') as f:
        pickle.dump(rmf, f)

def load(filename):
    global rmf
    with open(filename, 'rb') as f:
        rmf = pickle.load(f)