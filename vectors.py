"""
Title: vectors.py
Author: Oscar Chacon (orc2815@rit.edu)
"""

import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def main():
    with open('la_pf_10.json') as x:
        data = json.load(x)
    count_vect = CountVectorizer()
    test_counts = count_vect.fit_transform(data)
    test_counts.shape
    test_tfidf = tfidf_transformer.fit_transform(test_counts)
    test_tfidf.shape


main()
