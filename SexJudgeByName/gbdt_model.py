#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : gbdt_model.py
# @Author: Huangqinjian
# @Date  : 2018/4/5
# @Desc  :

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# 读取数据
train = pd.read_table('data/train.txt', ',')
test = pd.read_table('data/test.txt', ',')
submit = pd.read_csv('data/sample_submit.csv')

# 所有男生的名字
train_male = train[train['gender'] == 1]
m_cnt = len(train_male)
names_male = "".join(train_male['name'])

# 所有女生的名字
train_female = train[train['gender'] == 0]
f_cnt = len(train_female)
names_female = "".join(train_female['name'])

# 统计每个字在男生、女生名字中出现的总次数
# lists_male = map(lambda x: x.encode('utf-8'), names_male.decode('utf-8'))
# counts_male = Counter(lists_male)
# lists_female = map(lambda x: x.encode('utf-8'), names_female.decode('utf-8'))
# counts_female = Counter(lists_female)
lists_male = map(lambda x: x.encode('utf-8'), names_male)
counts_male = Counter(lists_male)
lists_female = map(lambda x: x.encode('utf-8'), names_female)
counts_female = Counter(lists_female)

# 得到训练集中每个人的每个字的词频（Term Frequency，通常简称TF）
train_encoded = []
for i in range(len(train)):
    name = train.at[i, 'name']
    # chs = map(lambda x: x.encode('utf-8'), name.decode('utf-8'))
    chs = list(map(lambda x: x.encode('utf-8'), name))
    row = [0., 0., 0., 0, train.at[i, 'gender']]
    for j in range(len(chs)):
        row[2 * j] = counts_female[chs[j]] * 1. / f_cnt
        row[2 * j + 1] = counts_male[chs[j]] * 1. / m_cnt
    train_encoded.append(row)

# 得到测试集中每个人的每个字的词频（Term Frequency，通常简称TF）
test_encoded = []
for i in range(len(test)):
    name = test.at[i, 'name']
    # chs = map(lambda x: x.encode('utf-8'), name.decode('utf-8'))
    chs = list(map(lambda x: x.encode('utf-8'), name))
    row = [0., 0., 0., 0., ]
    for j in range(len(chs)):
        try:
            row[2 * j] = counts_female[chs[j]] * 1. / f_cnt
        except:
            pass
        try:
            row[2 * j + 1] = counts_male[chs[j]] * 1. / m_cnt
        except:
            pass
    test_encoded.append(row)

# 转换为pandas.DataFrame的形式
# 1_f是指这个人的第一个字在训练集中所有女生的字中出现的频率
# 2_f是指这个人的第二个字在训练集中所有女生的字中出现的频率
# 1_m是指这个人的第一个字在训练集中所有男生的字中出现的频率
# 2_m是指这个人的第二个字在训练集中所有男生的字中出现的频率
train_encoded = pd.DataFrame(train_encoded, columns=['1_f', '1_m', '2_f', '2_m', 'gender'])
test_encoded = pd.DataFrame(test_encoded, columns=['1_f', '1_m', '2_f', '2_m'])

# 训练GBDT模型
# 85.30666666
# clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=300, subsample=0.5,
#                                  min_samples_split=6,
#                                  min_samples_leaf=3, max_depth=4, min_impurity_decrease=0,
#                                  warm_start=False)
# 85.3325
# clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=300, subsample=0.5,
#                                  min_samples_split=6,
#                                  min_samples_leaf=3, max_depth=4, min_impurity_decrease=0,
#                                  warm_start=True)
# 85.4025
# clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=400, subsample=0.5,
#                                  min_samples_split=6,
#                                  min_samples_leaf=3, max_depth=4, min_impurity_decrease=0,
#                                  warm_start=True)
# 85.469166666_500 85.53916666_600  85.58166666_700 85.67416_800  85.8825_1200 86.0216666_1500 86.1925_2000
clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=2500, subsample=0.5,
                                 min_samples_split=6,
                                 min_samples_leaf=3, max_depth=4, min_impurity_decrease=0,
                                 warm_start=True)
clf.fit(train_encoded.drop('gender', axis=1), train_encoded['gender'])
# print(clf.score(train_encoded.drop('gender', axis=1), train_encoded['gender']))
preds = clf.predict(test_encoded)

# 输出预测结果至my_TF_GBDT_prediction.csv
submit['gender'] = np.array(preds)
submit.to_csv('my_prediction_2.csv', index=False)
