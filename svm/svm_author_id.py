#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

# Import the relevant modules
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create a sklearn SVC classifier
clf = SVC(kernel = "rbf", C = 10000.)

# Slice the training dataset down to 1%, tossing out 99%.
# To speed up the algorithm. The tradeoff is that accuracy goes down.
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

# Train the classifier
timestamp = time() # Current time
clf.fit(features_train, labels_train)
print "Time taken for training:", round(time()-timestamp, 3), "s"

# Make a prediction using the classifier
timestamp = time() # Current time
pred = clf.predict(features_test)
print "Time taken for prediction:", round(time()-timestamp, 3), "s"

# Predicted elements: 10th, 26th, 50th
# Sara has label 0, Chris has label 1
print pred[10]
print pred[26]
print pred[50]

# Count values
count = 0
for e in pred:
    if e == 1:
        count += 1
print "How many Chris emails has predicted?:", count

# Estimate the accuracy
accuracy = accuracy_score(pred, labels_test)
print "Accuracy:", accuracy

#########################################################


