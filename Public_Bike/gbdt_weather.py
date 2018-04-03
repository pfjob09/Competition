#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : bike_value_modifyfeature.py
# @Author: Huangqinjian
# @Date  : 2018/3/28
# @Desc  :
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


def deal_train(trainFilePath):
    df = pd.read_csv(trainFilePath)
    data = df.copy(deep=True)

    # 离散型特征映射
    mapper = {'weather': {1: 4, 2: 3, 3: 2, 4: 1}}

    # 映射转换
    for col, mapItem in mapper.items():
        data.loc[:, col] = data[col].map(mapItem)

    data.to_csv('data/dealed_trainInput.csv', index=False)
    print(data.shape[0], data.shape[1])


def deal_test(testFilePath):
    df = pd.read_csv(testFilePath)
    data = df.copy(deep=True)

    # 离散型特征映射
    mapper = {'weather': {1: 4, 2: 3, 3: 2, 4: 1}}

    # 映射转换
    for col, mapItem in mapper.items():
        data.loc[:, col] = data[col].map(mapItem)

    data.to_csv('data/dealed_testInput.csv', index=False)
    print(data.shape[0], data.shape[1])


# 加载训练集
def load_trainData(trainFilePath):
    data = pd.read_csv(trainFilePath)
    X_trainList = []
    Y_trainList = []
    data_len = len(data)
    data_col = data.shape[1]
    for row in range(0, data_len):
        tmpList = []
        for col in range(1, data_col - 1):
            tmpList.append(data.iloc[row][col])
        X_trainList.append(tmpList)
        Y_trainList.append(data.iloc[row][data_col - 1])
    return X_trainList, Y_trainList


# 加载测试集
def load_testData(testFilePath):
    data = pd.read_csv(testFilePath)
    X_testList = []
    data_len = len(data)
    for row in range(0, data_len):
        tmpList = []
        for col in range(1, data.shape[1]):
            tmpList.append(data.iloc[row][col])
        X_testList.append(tmpList)
    return X_testList


def model_process(X_train, y_train, X_test):
    # GBDT训练过程

    model = GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=400, subsample=0.8,
                                      min_samples_split=3,
                                      min_samples_leaf=6, max_depth=5, warm_start=True)

    model.fit(X_train, y_train)
    # 对测试集进行预测
    ans = model.predict(X_test)

    ans_len = len(ans)
    id_list = np.arange(10001, 17001)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
    # print(pd_data)
    pd_data.to_csv('submit.csv', index=None)


if __name__ == '__main__':
    # deal_train('data/train.csv')
    # deal_test('data/test.csv')
    # 加载测试集，训练集
    X_train, Y_train = load_trainData('data/dealed_trainInput.csv')
    X_test = load_testData('data/dealed_testInput.csv')
    model_process(X_train, Y_train, X_test)
    # ====================================================================================================================
    # GBDT调试参数
    # cv_params = {'n_estimators': [100, 200, 300, 400, 500]}
    # other_params = {'loss': 'ls', 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.8,
    #                 'min_samples_split': 2,
    #                 'min_samples_leaf': 1, 'max_depth': 5, 'warm_start': True}

    # cv_params = {'max_depth': [4, 5, 6, 7, 8]}
    # other_params = {'loss': 'ls', 'learning_rate': 0.1, 'n_estimators': 200, 'subsample': 0.8,
    #                 'min_samples_split': 2,
    #                 'min_samples_leaf': 1, 'max_depth': 5, 'warm_start': True}

    # cv_params = {'subsample': [0.9, 0.95]}
    # other_params = {'loss': 'ls', 'learning_rate': 0.1, 'n_estimators': 200, 'subsample': 0.8,
    #                 'min_samples_split': 2,
    #                 'min_samples_leaf': 1, 'max_depth': 5, 'warm_start': True}

    # cv_params = {'min_samples_split': [2, 3, 4, 5, 6], 'min_samples_leaf': [2, 3, 4, 5, 6]}
    # other_params = {'loss': 'ls', 'learning_rate': 0.1, 'n_estimators': 200, 'subsample': 0.9,
    #                 'min_samples_split': 2,
    #                 'min_samples_leaf': 1, 'max_depth': 5, 'warm_start': True}

    # cv_params = {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 200, 300, 400, 500]}
    # other_params = {'loss': 'ls', 'learning_rate': 0.05, 'n_estimators': 400, 'subsample': 0.8,
    #                 'min_samples_split': 3,
    #                 'min_samples_leaf': 6, 'max_depth': 5, 'warm_start': True}
    #
    # model = GradientBoostingRegressor(**other_params)
    # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
    #                              verbose=1, n_jobs=4)
    # optimized_GBM.fit(X_train, Y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
