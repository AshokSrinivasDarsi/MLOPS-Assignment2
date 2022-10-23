import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import *

digits = datasets.load_digits()

n_samples = len(digits.images)
img = digits.images.reshape((n_samples, -1))
img.shape

xtrain, x_test, ytrain, y_test = train_test_split(img, digits.target, test_size=0.15, stratify=digits.target)

svmc = svm.SVC( kernel='rbf')
svmc.fit(xtrain, ytrain)

preds = svmc.predict(x_test)

def completely_biased_y_n(x):
    return pd.value_counts(x).shape[0]
def preds_all_classes_y_n(x):
    return pd.value_counts(x).shape[0]


import unittest

class TestNotebook(unittest.TestCase):
    
    def test_completely_biased_y_n(self):
        self.assertEqual(completely_biased_y_n(preds), 10)
        
    def test_preds_all_classes_y_n(self):
        self.assertEqual(preds_all_classes_y_n(preds), 10)

unittest.main(argv=[''], verbosity=2, exit=False)
