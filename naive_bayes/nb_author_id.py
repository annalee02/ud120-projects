#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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

# Import the relevant module:
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create an instance of a classifier
clf = GaussianNB()

# fit that instance, using the training data
timestamp = time() # Current time
clf.fit(features_train, labels_train)
print "Time taken for training:", round(time()-timestamp, 3), "s"

# Now I have a fitted model
# Make a prediction using the test data
timestamp = time() # Current time
pred = clf.predict(features_test)
print "Time taken for prediction:", round(time()-timestamp, 3), "s"

# Estimate the accuracy
print "Accuracy:", accuracy_score(pred, labels_test)

#########################################################


