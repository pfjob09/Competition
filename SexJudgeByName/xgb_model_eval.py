#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : xgb_model.py
# @Author: Huangqinjian
# @Date  : 2018/4/6
# @Desc  : https://github.com/HuangQinJian

import pandas as pd
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def load_data():
    # 读取数据
    train = pd.read_table('data/train.txt', ',')
    test = pd.read_table('data/test.txt', ',')

    # 所有男生的名字
    train_male = train[train['gender'] == 1]
    # print(train_male)
    m_cnt = len(train_male)

    names_male = "".join(train_male['name'])

    # 所有女生的名字
    train_female = train[train['gender'] == 0]
    f_cnt = len(train_female)
    names_female = "".join(train_female['name'])

    # 统计每个字在男生、女生名字中出现的总次数
    lists_male = map(lambda x: x.encode('utf-8'), names_male)
    counts_male = Counter(lists_male)
    lists_female = map(lambda x: x.encode('utf-8'), names_female)
    counts_female = Counter(lists_female)

    # 得到训练集中每个人的每个字的词频（Term Frequency，通常简称TF）
    train_encoded = []
    for i in range(len(train)):
        name = train.at[i, 'name']
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

    return train_encoded, test_encoded


def model_process(X_train, Y_train):
    X_train_split, X_test_split, Y_train_split, Y_test_split = train_test_split(X_train, Y_train, test_size=0.2)

    clf = xgb.XGBClassifier(learning_rate=0.1, n_estimators=10000, max_depth=5, min_child_weight=3, subsample=0.8,
                            colsample_bytree=0.6, gamma=0.1, reg_alpha=0, reg_lambda=1, metrics='error',
                            objective='binary:logistic')

    eval_set = [(X_train_split, Y_train_split), (X_test_split, Y_test_split)]

    clf.fit(X_train_split, Y_train_split, eval_set=eval_set, eval_metric='error', early_stopping_rounds=30)


def xgb_cv(X_train, Y_train):
    cv_params = {
        # 'n_estimators': range(500, 1000, 100)
        # 'max_depth': range(4, 10, 1),
        # 'min_child_weight': range(1, 6, 1),
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        # 'gamma': [0.3, 0.4, 0.5, 0.6],
        # 'reg_alpha': 0,
        # 'reg_lambda': 1,
        # 'learning_rate': 0.1
    }
    model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=500, max_depth=5, min_child_weight=4, seed=0,
                              subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_alpha=0, reg_lambda=1,
                              metrics='error',
                              objective='binary:logistic')
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='precision', cv=5,
                                 verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, Y_train)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


if __name__ == "__main__":
    train_encoded, test_encoded = load_data()

    # X_train = train_encoded.drop('gender', axis=1)
    # Y_train = train_encoded['gender']

    # xgb_cv(X_train, Y_train)
    # model_process(X_train, Y_train)
