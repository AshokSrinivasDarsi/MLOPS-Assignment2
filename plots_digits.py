

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import unittest
import argparse
import joblib
from sklearn.metrics import f1_score
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import *
from sklearn.tree import DecisionTreeClassifier


def build(args):
    print(args.clf_name,args.random_state)
    model = args.clf_name
    seed = int(args.random_state)

    digits = datasets.load_digits()

    n_samples = len(digits.images)
    img = digits.images.reshape((n_samples, -1))
    print(img.shape)

    xtrain, x_test, ytrain, y_test = train_test_split(img, digits.target, test_size=0.15, stratify=digits.target, random_state=seed)
    print(xtrain.shape,x_test.shape)

    xtrain, x_dev, ytrain, y_dev = train_test_split(xtrain, ytrain, test_size=0.15, stratify=ytrain, random_state=seed)
    print(xtrain.shape,x_test.shape)
    print(x_dev.shape,y_dev.shape)
    
    if model == 'svm':
    
        model = svm.SVC( kernel='rbf')
        model.fit(xtrain, ytrain)
        filename = '/Users/asrinivasdarsi/Desktop/Final_exam/models/{model}'.format(model=args.clf_name+'_gamma=0.001_C=0.2.joblib')

        
        
    else:
        model = DecisionTreeClassifier()
        model.fit(xtrain, ytrain)
        filename = '/Users/asrinivasdarsi/Desktop/Final_exam/models/{model}'.format(model=args.clf_name+'_criterion=gini.joblib')

        
    joblib.dump(model, filename)

    preds = model.predict(x_test)
    
    print('train accuracy',(accuracy_score(ytrain,model.predict(xtrain))))
    print('dev accuracy',(accuracy_score(y_dev,model.predict(x_dev))))
    print('test accuracy',(accuracy_score(y_test,model.predict(x_test))))
    print('test macro f1', f1_score(y_test, model.predict(x_test), average='macro'))
    
    lines = ['test accuracy '+':'+str((accuracy_score(y_test,model.predict(x_test)))),
            'test macro f1 '+':'+str(f1_score(y_test, model.predict(x_test), average='macro'))]
    with open('/Users/asrinivasdarsi/Desktop/Final_exam/results/{model}_{seed}.txt'.format(model=args.clf_name,seed=args.random_state), 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_name', default='svm') 
    parser.add_argument('--random_state' , default= 42) 
    args = parser.parse_args()
    
    build(args)

