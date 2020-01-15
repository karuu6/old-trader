from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from extract import label_samples
import numpy as np
from conf import *
import pickle
import json
import sys
import os

features = np.array(json.load(open('s_data/1_sec_features', 'r')))
prices   = np.array(json.load(open('s_data/1_sec_prices',   'r')))

#features, prices = extract_features()

x,y = label_samples(features, prices, order=10)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=128)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

with open('NOG', 'wb') as fp:
	pickle.dump(clf, fp)
