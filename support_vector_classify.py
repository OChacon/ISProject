from sklearn.svm import SVC
import pickle


vectormachine = SVC()


def trainSVM(vectorfile, resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    global vectormachine
    vectormachine = SVC()
    vectormachine.fit(vlist,rlist)


def testSVM(vectorfile, resultsfile):
    vlist = pickle.load(vectorfile)
    rlist = pickle.load(resultsfile)
    error = 0
    global vectormachine
    for x in range(len(vlist)):
        v = vlist[x]
        r = rlist[x]
        p = vectormachine.predict(v)
        error += abs(p-r)
    return error

def save(filename):
    with open(filename, 'wb+') as f:
        pickle.dump(vectormachine, f)

def load(filename):
    global vectormachine
    with open(filename, 'rb') as f:
        vectormachine = pickle.load(f)