#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : separate_train.py
# @Author: Huangqinjian
# @Date  : 2018/3/28
# @Desc  :
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


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
    id_list = []
    data_len = len(data)
    for row in range(0, data_len):
        tmpList = []
        id_list.append(data.iloc[row][0])
        for col in range(1, data.shape[1]):
            tmpList.append(data.iloc[row][col])
        X_testList.append(tmpList)
    return X_testList, id_list


def model_process(X_train, y_train, X_test, id_List):
    # GBDT训练过程   model_0
    # model = GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=250, subsample=0.6,
    #                                   min_samples_split=4,
    #                                   min_samples_leaf=5, max_depth=5, warm_start=True)
    # GBDT训练过程   model_1
    model = GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=300, subsample=0.6,
                                      min_samples_split=2,
                                      min_samples_leaf=5, max_depth=5, warm_start=True)

    model.fit(X_train, y_train)
    # 对测试集进行预测
    ans = model.predict(X_test)

    ans_len = len(ans)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([id_List[row], ans[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
    # print(pd_data)
    # pd_data.to_csv('submit_00.csv', index=None)
    pd_data.to_csv('submit_11.csv', index=None)


if __name__ == '__main__':
    # 加载测试集，训练集
    # X_train, Y_train = load_trainData('data/dealed_trainInput_0.csv')
    # X_test, id_List = load_testData('data/dealed_testInput_0.csv')
    X_train, Y_train = load_trainData('data/dealed_trainInput_1.csv')
    X_test, id_List = load_testData('data/dealed_testInput_1.csv')
    model_process(X_train, Y_train, X_test, id_List)
    # print(id_List)
