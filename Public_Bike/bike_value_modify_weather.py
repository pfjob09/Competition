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
from xgboost import plot_tree
from pandas import DataFrame
from xgboost import plot_importance
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
    # XGBoost训练过程
    model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=200, max_depth=6, min_child_weight=24, seed=0,
                             subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_alpha=0.05, reg_lambda=1,
                             metrics='rmse')

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

    # 显示重要特征
    # plot_importance(model)
    # plt.show()


if __name__ == '__main__':
    # deal_train('data/train.csv')
    # deal_test('data/test.csv')
    # 训练模型
    # 加载测试集，训练集
    X_train, Y_train = load_trainData('data/dealed_trainInput.csv')
    X_test = load_testData('data/dealed_testInput.csv')
    # 运行模型得到最终的submit.csv文件
    model_process(X_train, Y_train, X_test)
