#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:19:56 2020

@author: Khaled Nouri
"""

import pandas as pd
import numpy as np


 #User input   
urlData = './Iris.csv' #file contains the dataSet

label = 'Species' # the target attribute of the dataSet 
irrelevantAttributes = ['Id'] # irrelevant attribute, must be deleted before using the classifier
test =[5, 2.9, 0.77, 2] # tupe to predicat 


#Load/read the dataset from csv file
data = pd.read_csv(urlData)

#extract featurs from dataset
trainingSet = data.drop([label], axis = 1)
trainingSet = trainingSet.drop(irrelevantAttributes, axis = 1)
labels = np.array(data[label])


#Decision Tree Classifier 
from sklearn.tree import DecisionTreeClassifier as decisionTree
model = decisionTree(random_state=1)
model.fit(trainingSet, labels)
test = np.array([5.0, 3.6, 1.2, 0.17]).reshape(1, -1)
predicts = model.predict(test)

#print result 
print ("The label of " + str(test[0]) + " using decision tree is "  + str(predicts))