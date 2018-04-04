#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : bike_value.py
# @Author: Huangqinjian
# @Date  : 2018/3/28
# @Desc  :
import numpy as np
import pandas as pd
import xgboost as xgb
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
    data_len = len(data)
    for row in range(0, data_len):
        tmpList = []
        for col in range(1, data.shape[1]):
            tmpList.append(data.iloc[row][col])
        X_testList.append(tmpList)
    return X_testList


def model_process(X_train, y_train, X_test):
    # XGBoost训练过程
    model_xgb = xgb.XGBRegressor(learning_rate=0.07, n_estimators=200, max_depth=6, min_child_weight=6, seed=0,
                                 subsample=0.8, colsample_bytree=0.8, gamma=0.3, reg_alpha=0.05, reg_lambda=3,
                                 metrics='rmse')
    model_xgb.fit(X_train, y_train)

    # 对测试集进行预测
    predict_xgb = model_xgb.predict(X_test)

    # GBDT训练过程
    mode_gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=200, subsample=0.6,
                                          min_samples_split=3,
                                          min_samples_leaf=5, max_depth=6, warm_start=True)

    mode_gbdt.fit(X_train, y_train)
    # 对测试集进行预测
    predict_gbdt = mode_gbdt.predict(X_test)
    # 融合两个模型
    predict_len = len(predict_xgb)
    predict = []
    temp = 0
    for i in range(0, predict_len):
        # score:14.879
        # temp = 0.6 * predict_gbdt[i] + 0.4 * predict_xgb[i]
        # 14.892
        # temp = 0.7 * predict_gbdt[i] + 0.3 * predict_xgb[i]
        # 14.878
        temp = 0.65 * predict_gbdt[i] + 0.35 * predict_xgb[i]
        predict.append(temp)

    id_list = np.arange(10001, 17001)
    data_arr = []
    for row in range(0, predict_len):
        data_arr.append([int(id_list[row]), predict[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
    # print(pd_data)
    pd_data.to_csv('submit.csv', index=None)


if __name__ == '__main__':
    # 训练模型
    # 加载测试集，训练集
    X_train, Y_train = load_trainData('data/train.csv')
    X_test = load_testData('data/test.csv')
    # 运行模型得到最终的submit.csv文件
    model_process(X_train, Y_train, X_test)
