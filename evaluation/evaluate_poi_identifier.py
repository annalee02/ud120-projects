#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
sort_keys = '../tools/python2_lesson14_keys.pkl'

### POI identifier from validate_poi.py
from sklearn import cross_validation
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

print len(data)
# hold out 30% of the data for testing
# random_state controls which points go into the training set and which are used for testing
d_train, d_test, l_train, l_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
print len(d_train)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
pred = clf.predict(features)

print "score: "
print clf.score(features,labels)
print accuracy_score(pred, labels)

# fit the data on the training set
clf_subset = clf.fit(d_train, l_train)
pred2 = clf_subset.predict(d_test)
print "Accuracy score on the test set"
print accuracy_score(pred2, l_test)
print "clf score on the test set"
print clf.score(d_test,l_test)

### my code goes here
count = 0
# poi is the placeholder for each item in 'l_test'
# print l_test[:10]
for poi in l_test:
    if poi == 1.0:
        count +=1
print "How many POIs are predicted for the test set?", count

print "How many people total are in the test set?", len(d_test)

no_tp = np.sum([1 for j in zip(l_test, pred2) if j[0] == j[1] and j[1] == 1])

print "Number of True Positives:", no_tp

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "precision score:", precision_score(l_test, pred2)
print "recall score:", recall_score(l_test, pred2)

###############
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

no_tp = np.sum([1 for j in zip(true_labels, predictions) if j[0] == j[1] and j[1] == 1])
print "How many True Positives?", no_tp
no_tn = np.sum([1 for j in zip(true_labels, predictions) if j[0] == j[1] and j[1] != 1])
print "How many True Negatives?", no_tn
no_fp = np.sum([1 for j in zip(true_labels, predictions) if j[0] != j[1] and j[1] == 1])
print "How many False Positives?", no_fp
no_fn = np.sum([1 for j in zip(true_labels, predictions) if j[0] != j[1] and j[1] != 1])
print "How many False Negatives?", no_fn

print "precision score:", precision_score(true_labels, predictions)
print "recall score:", recall_score(true_labels, predictions)
