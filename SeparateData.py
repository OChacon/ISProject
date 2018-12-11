import json

"""
A script to Seperate the json file into three files of size 50/25/25 and all with an equal number from each
Subreddit.
The files are then saved as
trainData.json - For Training
testData.json - For testing and development purposes
evalData.json - For final evalutation

Author: Mike Hurlbutt
"""
def segrigate(filename):
    fp = open(filename)
    data = json.load(fp)
    total = len(data)
    trainlen = total/2
    otherlen = total/4
    train = {}
    traincounts = [0,0]
    test = {}
    testcounts = [0,0]
    eval = {}
    evalcounts = [0,0]
    subs = []
    for d in data:
        if data[d]['subReddit'] not in subs:
            subs.append(data[d]['subReddit'])
        if data[d]['subReddit'] == subs[0]:
            if traincounts[0] != int(trainlen/2):
                train[d] = data[d]
                traincounts[0] += 1
            elif testcounts[0] != int(otherlen/2):
                test[d] = data[d]
                testcounts[0] += 1
            else:
                eval[d] = data[d]
                evalcounts[0] += 1
        else:
            if traincounts[1] != int(trainlen/2):
                train[d] = data[d]
                traincounts[1] += 1
            elif testcounts[1] != int(otherlen/2):
                test[d] = data[d]
                testcounts[1] += 1
            else:
                eval[d] = data[d]
                evalcounts[1] += 1
    of = open('trainData.json', 'w+')
    json.dump(train, of, indent=4, separators=(',', ': '))
    of = open('testData.json', 'w+')
    json.dump(test, of, indent=4, separators=(',', ': '))
    of = open('evalData.json', 'w+')
    json.dump(eval, of, indent=4, separators=(',', ': '))

if __name__ == '__main__':
    segrigate('la_pf_100.json')