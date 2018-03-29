#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : soccer_value.py
# @Author: Huangqinjian
# @Date  : 2018/3/22
# @Desc  :

import numpy as np
import pandas as pd
import xgboost as xgb
import pydot
import operator
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import plot_tree
from xgboost import plot_importance
from graphviz import Digraph


# 处理训练集的数据，进行特征处理，输出经过处理过的特征的训练集文件
def deal_train(trainFilePath):
    data = pd.read_csv(trainFilePath)

    data_deal = data.copy(deep=True)

    # 离散型特征映射
    mapper = {'work_rate_att': {'Low': 1, 'Medium': 2, 'High': 3}, 'work_rate_def': {'Low': 1, 'Medium': 2, 'High': 3}}

    # 对结果影响很小,或者与其他特征相关性较高的特征将被丢弃
    dropList = ['birth_date', 'height_cm', 'weight_kg', 'gk']

    # 映射转换
    for col, mapItem in mapper.items():
        data_deal.loc[:, col] = data_deal[col].map(mapItem)

    # 构造新的特征BIM
    data_deal['bim'] = data_deal['weight_kg'] / ((data_deal['height_cm'] / 100) ** 2)

    # 丢弃特征
    data_deal.drop(dropList, axis=1, inplace=True)

    # 处理有缺失值的特征，离散型特征填充众数,数值型特征填充平均数
    na_col = data_deal.dtypes[data_deal.isnull().any()]
    for col in na_col.index:
        if na_col[col] != 'object':
            med = data_deal[col].mean()
            data_deal[col].fillna(med, inplace=True)
        else:
            mode = data_deal[col].mode()[0]
            data_deal[col].fillna(mode, inplace=True)

    # 输出经过处理过的特征的训练集文件
    data_deal.to_csv('dataset/soccer/dealed_trainInput.csv', index=False)
    print(data_deal.shape[0], data_deal.shape[1])


# 处理测试集的数据，进行特征处理，输出经过处理过的特征的测试集文件
def deal_test(testFilePath):
    data = pd.read_csv(testFilePath)

    data_deal = data.copy(deep=True)

    # 离散型特征映射
    mapper = {'work_rate_att': {'Low': 1, 'Medium': 2, 'High': 3}, 'work_rate_def': {'Low': 1, 'Medium': 2, 'High': 3}}

    # 对结果影响很小,或者与其他特征相关性较高的特征将被丢弃
    dropList = ['birth_date', 'height_cm', 'weight_kg', 'gk']

    # 映射转换
    for col, mapItem in mapper.items():
        data_deal.loc[:, col] = data_deal[col].map(mapItem)

    # 构造新的特征BIM
    data_deal['bim'] = data_deal['weight_kg'] / ((data_deal['height_cm'] / 100) ** 2)

    # 丢弃特征
    data_deal.drop(dropList, axis=1, inplace=True)

    # 处理有缺失值的特征，离散型特征填充众数,数值型特征填充平均数
    na_col = data_deal.dtypes[data_deal.isnull().any()]
    for col in na_col.index:
        if na_col[col] != 'object':
            med = data_deal[col].mean()
            data_deal[col].fillna(med, inplace=True)
        else:
            mode = data_deal[col].mode()[0]
            data_deal[col].fillna(mode, inplace=True)

    # 输出经过处理过的特征的测试集文件
    data_deal.to_csv('dataset/soccer/dealed_testInput.csv', index=False)
    print(data_deal.shape[0], data_deal.shape[1])


# 加载训练集
def load_trainData(trainFilePath):
    data = pd.read_csv(trainFilePath)
    X_trainList = []
    Y_trainList = []
    data_len = len(data)
    data_col = data.shape[1]
    for row in range(0, data_len):
        tmpList = []
        for col in range(1, data_col - 2):
            tmpList.append(data.iloc[row][col])
        tmpList.append(data.iloc[row][data_col - 1])
        X_trainList.append(tmpList)
        Y_trainList.append(data.iloc[row][data_col - 2])
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
    model = xgb.XGBRegressor(learning_rate=0.05, n_estimators=2000, max_depth=9, min_child_weight=5, seed=0,
                             subsample=0.8, colsample_bytree=0.8, gamma=0.2, reg_alpha=0, reg_lambda=1, metrics='mae')
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


def mergeFeature(Feature, new_Feature):
    tmp_Feature = []
    for row in range(0, len(Feature)):
        tmp = np.array([list(Feature[row]), list(new_Feature[row])])
        tmp_Feature.append(list(np.hstack(tmp)))
    return tmp_Feature


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i += 1
    outfile.close()


if __name__ == '__main__':
    # #得到经过处理过的特征的训练集文件
    # deal_train('dataset/soccer/train.csv')
    # #得到经过处理过的特征的测试集文件
    # deal_test('dataset/soccer/test.csv')
    # =============================================================================================
    # MAE：34.32   17
    # 训练模型
    # #加载测试集，训练集
    # X_train, Y_train = load_trainData('dataset/soccer/dealed_trainInput.csv')
    # X_test = load_testData('dataset/soccer/dealed_testInput.csv')
    # print(len(X_train))
    # print(len(Y_train))
    # print(Y_train[:5])
    # print(len(X_test))
    # 运行模型得到最终的submit.csv文件
    # model_process(X_train, Y_train, X_test)
    # =============================================================================================
    # MAE：36.7102   17
    # 利用XGBoost得到新的特征训练模型
    # 加载测试集，训练集
    # X_train, Y_train = load_trainData('dataset/soccer/dealed_trainInput.csv')
    # X_test = load_testData('dataset/soccer/dealed_testInput.csv')
    # model = xgb.XGBRegressor(learning_rate=0.05, n_estimators=2000, max_depth=9, min_child_weight=5, seed=0,
    #                          subsample=0.8, colsample_bytree=0.8, gamma=0.2, reg_alpha=0, reg_lambda=1, metrics='mae')
    # model.fit(X_train, Y_train)
    # # 得到新的训练集的特征
    # new_X_train = model.apply(X_train)
    # print(new_X_train.shape)
    # # 得到新的测试集的特征
    # new_X_test = model.apply(X_test)
    # print(new_X_test.shape)
    # # 得到的新特征DataFrame化
    # new_train_feature = DataFrame(new_X_train)
    # new_test_feature = DataFrame(new_X_test)
    # # 运行模型得到最终的submit.csv文件
    # model_process(new_train_feature, Y_train, new_test_feature)
    # =============================================================================================
    # 利用XGBoost得到新的特征，与原特征合并后训练模型
    # 加载测试集，训练集
    """
    X_train, Y_train = load_trainData('dataset/soccer/dealed_trainInput.csv')
    X_test = load_testData('dataset/soccer/dealed_testInput.csv')
    model = xgb.XGBRegressor(learning_rate=0.05, n_estimators=2000, max_depth=9, min_child_weight=5, seed=0,
                             subsample=0.8, colsample_bytree=0.8, gamma=0.2, reg_alpha=0, reg_lambda=1, metrics='mae')
    model.fit(X_train, Y_train)
    # 得到新的训练集的特征
    new_X_train = model.apply(X_train)
    print(new_X_train.shape)
    # 得到新的测试集的特征
    new_X_test = model.apply(X_test)
    print(new_X_test.shape)
    # 得到的新特征DataFrame化
    new_train_feature = DataFrame(new_X_train)
    new_test_feature = DataFrame(new_X_test)
    new_train_feature.to_csv('dataset/soccer/new_train_feature.csv', index=False)
    new_test_feature.to_csv('dataset/soccer/new_test_feature.csv', index=False)
    print("新特征处理完毕")
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`~~~~~~
    # 新特征与原有的特征进行合并
    # X_train_new = mergeFeature(X_train, new_X_train)
    # X_test_new = mergeFeature(X_test, new_X_test)
    # print("特征合并完毕")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`~~~~~~
    # 把合并后的特征保存到文件中
    # df_train_old = pd.read_csv('dataset/soccer/dealed_trainInput.csv')
    # df_test_old = pd.read_csv('dataset/soccer/dealed_testInput.csv')
    #
    # df_train_new = pd.read_csv('dataset/soccer/new_train_feature.csv')
    # df_test_new = pd.read_csv('dataset/soccer/new_test_feature.csv')
    #
    # df_merged_train = pd.concat([df_train_new, df_train_old], axis=1, ignore_index=True)
    # df_merged_test = pd.concat([df_test_new, df_test_old], axis=1, ignore_index=True)
    #
    # df_merged_train.to_csv('dataset/soccer/merged_train_feature.csv', index=True)
    # df_merged_test.to_csv('dataset/soccer/merged_test_feature.csv', index=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`~~~~~~
    # 运行模型得到最终的submit.csv文件
    X_train, Y_train = load_trainData('dataset/soccer/merged_train_feature.csv')
    X_test = load_testData('dataset/soccer/merged_test_feature.csv')
    model_process(X_train, Y_train, X_test)
    # =============================================================================================
    # XGBoost调试参数

    # 加载测试集，训练集
    # X_train, Y_train = load_trainData('dataset/soccer/dealed_trainInput.csv')

    # cv_params = {'n_estimators': [1100, 1200, 1300, 1500, 2000]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 2000, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    # cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 2000, 'max_depth': 9, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 2000, 'max_depth': 9, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 2000, 'max_depth': 9, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}

    # cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    # other_params = {'learning_rate': 0.05, 'n_estimators': 2000, 'max_depth': 9, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_error', cv=5,
    #                              verbose=1, n_jobs=4)
    # optimized_GBM.fit(X_train, Y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    # =============================================================================================
    # Plotting API画图，查看每棵树的结构
    # 加载测试集，训练集
    # X_train, Y_train = load_trainData('dataset/soccer/dealed_trainInput.csv')
    # # 获取训练集的列名，即特征的名字
    # df = pd.read_csv('dataset/soccer/dealed_trainInput.csv')
    # features = df.columns.tolist()
    # ceate_feature_map(features)
    # # XGBoost训练过程
    # model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=500, max_depth=2, min_child_weight=5, seed=0,
    #                          subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=1, reg_lambda=1, metrics='mae')
    # model.fit(X_train, Y_train)
    # plot_tree(model, fmap='xgb.fmap', num_trees=5, rankdir='LR')
    # plot_importance(model)
    # plt.show()
