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
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf=DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

############################################
#accuracy 
acc = accuracy_score(pred, labels_test)
print("Accuracy: ", acc) # accuracy with percentile=10 #0.9772468714448237

######################################
#What's the number of features in your data?
num_features = len(features_train[0])
print("Number of features:", num_features) #3785 #percentile=10

#########################################
#selector = SelectPercentile(f_classif, percentile=1) #379

###############################################
## accuracy with percentile=1 # 0.9664391353811149
