import reddit_download
import vectors
import SeeeegrigationTimeCmon as seg
import support_vector_classify as svc
import random_forest_classify as rfc
import LSTM_classify as lstm
import os
import argparse

def download_Data_Vector():
    # Training Data
    reddit_download.download('legaladvice', 'personalfinance', 'la_pf_100.json', 100)
    seg.segrigate('la_pf_100.json')
    vectors.vectorize(['trainData.json', 'testData.json', 'evalData.json'], [100, 50, 50])
    vectors.vectorize(['trainData.json', 'testData.json', 'evalData.json'], [100, 50, 50], 200)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",action='store_true',help='Present if data download is required')
    parser.add_argument("-algs",nargs='+',help="The one or more algorithms to run SVC, LSTM, and/or RFC")
    parser.add_argument("-train",action='store_true',help='If you want to train and print the error of the testdata of the algs')
    parser.add_argument("-eval",action='store_true',help='If you want to print the error of the algs using the Evaluation Data')
    parser.add_argument("-load", action='store_true',
                        help='If you want reload existing models for evaluation or training for the lstm')
    args = parser.parse_args()

    if args.d:
        download_Data_Vector()
    if 'SVC' in args.algs:
        print("Support Vector Classifier:")
        if args.train:
            svc.trainSVM('trainData.vvec','trainData.rvec')
            print("Accuracy: "+ str(1-svc.testSVM('testData.vvec','testData.rvec')/50.0))
        if args.eval:
            if args.load:
                svc.load('svm.mdl')
            print("Evaluation Accuracy:")
            print(str(1-svc.testSVM('evalData.vvec','evalData.rvec')/50.0))
    if 'RFC' in args.algs:
        print("Random Forest Classifier:")
        if args.train:
            rfc.trainRMF('trainData.vvec','trainData.rvec')
            print("Accuracy: "+ str(1-rfc.testRMF('testData.vvec','testData.rvec')/50.0))
        if args.eval:
            if args.load:
                rfc.load('rmf.mdl')
            print("Evaluation Accuracy:")
            print(str(1-rfc.testRMF('evalData.vvec','evalData.rvec')/50.0))
    if 'LSTM' in args.algs:
        print("Long Short Term Memory Classifier:")
        if args.train:
            if args.load:
                lstm.trainLSTM('trainData200.vvec', 'trainData200.rvec','lstm.mdl')
            else:
                lstm.trainLSTM('trainData200.vvec', 'trainData200.rvec')
            print('Accuracy: '+ str(1-lstm.testLSTM('testData200.vvec', 'testData200.rvec')/50))
        if args.eval:
            print("Evalution Accuracy:")
            if args.load:
                print(str(1-lstm.testLSTM('evalData200.vvec', 'evalData200.rvec', 'lstm.mdl') / 50))
            else:
                print(str(1 - lstm.testLSTM('evalData200.vvec', 'evalData200.rvec') / 50))

if __name__ == '__main__':
    main()