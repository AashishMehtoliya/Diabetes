# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:37:18 2019

@author: Aashish Mehtoliya
"""

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = loadtxt('pima-indians-diabetes.csv', delimiter=",")

X = df[:,0:8]
Y = df[:,8]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_seed = 23)

model = XGBClassifier()
model.fit(X_train,y_train)

print(model)

y_pred = model.predict(X_test)
predictions = [round(i) for i in y_pred]

accuracy = accuracy_score(y_test,predictions)

print(accuracy*100.0)




