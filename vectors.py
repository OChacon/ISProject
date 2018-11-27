"""
Title: vectors.py
Author: Oscar Chacon (orc2815@rit.edu)
"""

import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import pickle
import os

def vectorize(filename):
    resultList = []
    with open(filename) as x:
        data = json.load(x)
    labels = []
    texts = []
    for d in data:
        if data[d]['subReddit'] not in labels:
            labels.append(data[d]['subReddit'])
        resultList.append(labels.index(data[d]['subReddit']))
        texts.append(data[d]['title'] + '\n' + data[d]['desc'] + '\n' + data[d]['topComment'])
    count_vect = CountVectorizer()
    test_counts = count_vect.fit_transform(texts)
    test_counts.shape
    tfidf_transformer = TfidfTransformer()
    test_tfidf = tfidf_transformer.fit_transform(test_counts)
    test_tfidf.shape
    with open(os.path.splitext(filename)[0]+".rvec", 'wb+') as f:
        pickle.dump(np.array(resultList),f)
    with open(os.path.splitext(filename)[0]+".vvec", 'wb+') as f:
        pickle.dump(test_tfidf,f)
    return test_tfidf, np.array(resultList)


if __name__ == '__main__':
    vectorize('la_pf_10.json')
