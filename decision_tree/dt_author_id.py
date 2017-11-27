#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### my code goes here ###

# What's the number of features in the data?
# It is organized into a numpy array
print "The number of features in the data:", len(features_train[0])

# Import the relevant modules
from sklearn import tree
from sklearn.metrics import accuracy_score

# Create a sklearn DT classifier
# Set min_samples_split=40
clf = tree.DecisionTreeClassifier(min_samples_split=40)

# Train the classifier
clf = clf.fit(features_train, labels_train)

# Make a prediction using the classifier
pred = clf.predict(features_test)

acc = accuracy_score(pred, labels_test)

# Estimate the accuracy
def submitAccuracies():
    return {"acc":round(acc,3)}
print submitAccuracies()

# Quiz:
# Go into ../tools/email_preprocess.py, and find the line of code that looks like this:
# selector = SelectPercentile(f_classif, percentile=10)
# Change percentile from 10 to 1, and re-run dt_author_id.py. 
# What's the accuracy of the decision tree when I use only 1% of my available features (i.e. percentile=1)?
# Result = 0.966
# It was 0.976 when using percentile=10
#########################################################
