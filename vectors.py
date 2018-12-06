"""
Title: vectors.py
Author: Oscar Chacon (orc2815@rit.edu)
"""

import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import pickle
import os

def vectorize(filenames,counts,m=-1):
    resultList = []
    texts = []
    labels = []
    for filename in filenames:
        with open(filename) as x:
            data = json.load(x)
        for d in data:
            if data[d]['subReddit'] not in labels:
                labels.append(data[d]['subReddit'])
            resultList.append(labels.index(data[d]['subReddit']))
            texts.append(data[d]['title'] + '\n' + data[d]['desc'] + '\n' + data[d]['topComment'])
    if m != -1:
        count_vect = CountVectorizer(stop_words='english',max_features=m)
    else:
        count_vect = CountVectorizer(stop_words='english')
    test_counts = count_vect.fit_transform(texts)
    test_counts.shape
    tfidf_transformer = TfidfTransformer()
    test_tfidf = tfidf_transformer.fit_transform(test_counts)
    test_tfidf.shape

    slices = []
    for x in range(len(filenames)):
        filename = filenames[x]
        count = 0
        for y in range(0,x):
            count += counts[y]
        slices.append((count,count+counts[x]))
        if m != -1:
            with open(os.path.splitext(filename)[0]+str(m) + ".rvec", 'wb+') as f:
                pickle.dump(np.array(resultList[count:count+counts[x]]),f)
            with open(os.path.splitext(filename)[0]+str(m) +".vvec", 'wb+') as f:
                pickle.dump(test_tfidf[count:count+counts[x]],f)
        else:
            with open(os.path.splitext(filename)[0]+".rvec", 'wb+') as f:
                pickle.dump(np.array(resultList[count:count+counts[x]]),f)
            with open(os.path.splitext(filename)[0]+".vvec", 'wb+') as f:
                pickle.dump(test_tfidf[count:count+counts[x]],f)
    return test_tfidf, np.array(resultList)


if __name__ == '__main__':
    vectorize(['trainData.json','testData.json','evalData.json'],[100,50,50])
    vectorize(['trainData.json','testData.json','evalData.json'], [100, 50, 50], 200)
