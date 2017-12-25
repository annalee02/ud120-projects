#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
sort_keys = '../tools/python2_lesson13_keys.pkl'

### it's all yours from here forward!
from sklearn import cross_validation
from sklearn import tree
from sklearn.metrics import accuracy_score

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
