#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : KNN_Solve.py
# @Author: Huangqinjian
# @Date  : 2018/4/4
# @Desc  :
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

train_labels = train.pop('y')

clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(train, train_labels)
print(clf.score(train, train_labels))

submit = pd.read_csv('data/sample_submit.csv')
submit['y'] = clf.predict(test)
submit.to_csv('knn2.csv', index=False)
