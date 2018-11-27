from sklearn.svm import SVC
import numpy as np
import pickle


vectormachine = SVC()


def trainSVM(vectorfile, resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    global vectormachine
    vectormachine = SVC(gamma='auto')
    vectormachine.fit(vlist,rlist)


def testSVM(vectorfile, resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    error = 0
    global vectormachine
    error = sum(np.abs(vectormachine.predict(vlist)-rlist))
    #for x in range(vlist.shape[0]):
    #    v = vlist[x]
    #    r = rlist[x]
    #    p = vectormachine.predict(v)
    #    error += abs(p-r)
    return error

def save(filename):
    global vectormachine
    with open(filename, 'wb+') as f:
        pickle.dump(vectormachine, f)

def load(filename):
    global vectormachine
    with open(filename, 'rb') as f:
        vectormachine = pickle.load(f)

if __name__ == '__main__':
    trainSVM('la_pf_10.vvec','la_pf_10.rvec')
    print(testSVM('la_pf_10.vvec','la_pf_10.rvec'))
    save('svm.mdl')