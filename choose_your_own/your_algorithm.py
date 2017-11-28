#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

# Import the relevant modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a sklearn RandomForest classifier
clf = RandomForestClassifier(max_depth=20)

# Train the classifier
clf = clf.fit(features_train, labels_train)

# Make a prediction using the classifier
pred = clf.predict(features_test)

# Estimate the accuracy
accuracy = accuracy_score(pred, labels_test)
print "Accuracy:", accuracy

################################################################################

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

# Show the decision boundary
plt.show()
