
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

digits=load_digits()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(digits.data, digits.target, test_size=0.3)

lr=LogisticRegression()
lr.fit(x_train, y_train)
lr.score(x_test, y_test)

svm=SVC()
svm.fit(x_train, y_train)
svm.score(x_test, y_test)

rf=RandomForestClassifier()
rf.fit(x_train, y_train)
rf.score(x_test, y_test)

"""Evaluating model using cross val score to find that which model works best for the given data set"""

from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression() , digits.data, digits.target)

cross_val_score(SVC() , digits.data, digits.target)

cross_val_score(RandomForestClassifier() , digits.data, digits.target)

"""Hyperparameter optimization and tuning of machine learning"""

cross_val_score(RandomForestClassifier(n_estimators=10) , digits.data, digits.target)

cross_val_score(RandomForestClassifier(n_estimators=30) , digits.data, digits.target)

cross_val_score(RandomForestClassifier(n_estimators=40) , digits.data, digits.target)

cross_val_score(RandomForestClassifier(n_estimators=60) , digits.data, digits.target)
