#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : soccer_value.py
# @Author: Huangqinjian
# @Date  : 2018/3/22
# @Desc  :

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import preprocessing
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split


def featureSet(data):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer.fit(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])
    x_new = imputer.transform(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])

    le = preprocessing.LabelEncoder()
    le.fit(['Low', 'Medium', 'High'])
    att_label = le.transform(data.work_rate_att.values)
    # print(att_label)
    def_label = le.transform(data.work_rate_def.values)
    # print(def_label)

    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['club'])
        tmp_list.append(data.iloc[row]['league'])
        tmp_list.append(data.iloc[row]['potential'])
        tmp_list.append(data.iloc[row]['international_reputation'])
        tmp_list.append(data.iloc[row]['pac'])
        tmp_list.append(data.iloc[row]['sho'])
        tmp_list.append(data.iloc[row]['pas'])
        tmp_list.append(data.iloc[row]['dri'])
        tmp_list.append(data.iloc[row]['def'])
        tmp_list.append(data.iloc[row]['phy'])
        tmp_list.append(data.iloc[row]['skill_moves'])
        tmp_list.append(x_new[row][0])
        tmp_list.append(x_new[row][1])
        tmp_list.append(x_new[row][2])
        tmp_list.append(x_new[row][3])
        tmp_list.append(x_new[row][4])
        tmp_list.append(x_new[row][5])
        tmp_list.append(att_label[row])
        tmp_list.append(def_label[row])
        XList.append(tmp_list)
    yList = data.y.values
    return XList, yList


def loadTestData(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer.fit(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])
    x_new = imputer.transform(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])

    le = preprocessing.LabelEncoder()
    le.fit(['Low', 'Medium', 'High'])
    att_label = le.transform(data.work_rate_att.values)
    # print(att_label)
    def_label = le.transform(data.work_rate_def.values)
    # print(def_label)

    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['club'])
        tmp_list.append(data.iloc[row]['league'])
        tmp_list.append(data.iloc[row]['potential'])
        tmp_list.append(data.iloc[row]['international_reputation'])
        tmp_list.append(data.iloc[row]['pac'])
        tmp_list.append(data.iloc[row]['sho'])
        tmp_list.append(data.iloc[row]['pas'])
        tmp_list.append(data.iloc[row]['dri'])
        tmp_list.append(data.iloc[row]['def'])
        tmp_list.append(data.iloc[row]['phy'])
        tmp_list.append(data.iloc[row]['skill_moves'])
        tmp_list.append(x_new[row][0])
        tmp_list.append(x_new[row][1])
        tmp_list.append(x_new[row][2])
        tmp_list.append(x_new[row][3])
        tmp_list.append(x_new[row][4])
        tmp_list.append(x_new[row][5])
        tmp_list.append(att_label[row])
        tmp_list.append(def_label[row])
        XList.append(tmp_list)
    return XList


def trainandTest(X_train, y_train, X_test):
    # XGBoost训练过程
    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=500, silent=False, objective='reg:gamma')
    model.fit(X_train, y_train)

    # 对测试集进行预测
    ans = model.predict(X_test)

    ans_len = len(ans)
    id_list = np.arange(10441, 17441)
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


def trainandTestBasic(X_train, y_train, X_test):
    params = {
        'booster': 'gbtree',
        'objective': 'reg:gamma',
        'gamma': 0.1,
        'max_depth': 5,
        'lambda': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(X_train, y_train)
    num_rounds = 300
    plst = params.items()
    model = xgb.train(plst, dtrain, num_rounds)

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test)
    ans = model.predict(dtest)

    ans_len = len(ans)
    id_list = np.arange(10441, 17441)
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
    trainFilePath = 'dataset/soccer/train.csv'
    testFilePath = 'dataset/soccer/test.csv'
    data = pd.read_csv(trainFilePath)
    X_train, y_train = featureSet(data)
    X_test = loadTestData(testFilePath)
    trainandTest(X_train, y_train, X_test)
    """
    params = {
        'booster': 'gbtree',
        'objective': 'reg:gamma',
        'eta': 0.1,
        'max_depth': 10,
        'subsample': 1.0,
        'min_child_weight': 5,
        'colsample_bytree': 0.2,
        'scale_pos_weight': 0.1,
        'gamma': 0.2,
        'lambda': 300
    }
    train_X, test_X, train_y, test_y = train_test_split(data.iloc[:, -2:],
                                                        data.iloc[:, -1],
                                                        test_size=0.3,
                                                        random_state=0)
    dtrain = xgb.DMatrix(train_X, train_y)
    dval = xgb.DMatrix(test_X, test_y)
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(params, dtrain, num_boost_round=100000, evals=watchlist)
    """
