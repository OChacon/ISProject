import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

rmf = RandomForestClassifier(n_estimators=100,max_depth=4)

def trainRMF(vectorfile, resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    global rmf
    rmf = RandomForestClassifier(n_estimators=100,max_depth=4)
    rmf.fit(vlist,rlist)

def testRMF(vectorfile, resultsfile):
    with open(vectorfile, 'rb') as f:
        vlist = pickle.load(f)
    with open(resultsfile, 'rb') as f:
        rlist = pickle.load(f)
    error = 0
    global rmf
    error = sum(np.abs(rmf.predict(vlist) - rlist))
    #for x in range(len(vlist)):
    ##    r = rlist[x]
     #   p = rmf.predict(v)
      #  error += abs(p-r)
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
    trainRMF('la_pf_10.vvec','la_pf_10.rvec')
    print(testRMF('la_pf_10.vvec','la_pf_10.rvec'))
    save('rmf.mdl')