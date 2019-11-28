
'''
Download clean_df
'''

import pandas as pd
import numpy as np
import os
import datetime
import time
import glob
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import sklearn.svm as svm
import pickle


clean_df = pd.read_csv("../Data/clean_df.csv", encoding='utf-8')

X = clean_df.loc[:, clean_df.columns != "DAYS_TO_PAY"]
y = clean_df['DAYS_TO_PAY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

svr = svm.SVR(kernel='poly', C=100, gamma=0.1, epsilon=.1)
clf = svr.fit(X_train, y_train)

model_filename = "svr_poly.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(clf, model_file)

