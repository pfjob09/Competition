#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : xgb_knn_model.py
# @Author: Huangqinjian
# @Date  : 2018/4/6
# @Desc  :

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor


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
    data_deal.to_csv('data/dealed_trainInput.csv', index=False)
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
    data_deal.to_csv('data/dealed_testInput.csv', index=False)
    print(data_deal.shape[0], data_deal.shape[1])


# 加载训练集
def load_trainData(trainFilePath):
    data = pd.read_csv(trainFilePath)
    train_data_xgb = data.copy()
    dropList = ['id', 'club', 'league', 'nationality', 'international_reputation', 'skill_moves', 'weak_foot',
                'work_rate_att',
                'work_rate_def', 'preferred_foot', 'y']
    train_data_xgb.drop(dropList, axis=1, inplace=True)
    col_knn = ['club', 'league', 'nationality', 'international_reputation', 'skill_moves', 'weak_foot', 'work_rate_att',
               'work_rate_def', 'preferred_foot']
    return train_data_xgb, data[col_knn], data['y']


# 加载测试集
def load_testData(testFilePath):
    data = pd.read_csv(testFilePath)
    test_data_xgb = data.copy()
    dropList = ['id', 'club', 'league', 'nationality', 'international_reputation', 'skill_moves', 'weak_foot',
                'work_rate_att',
                'work_rate_def', 'preferred_foot']
    test_data_xgb.drop(dropList, axis=1, inplace=True)
    col_knn = ['club', 'league', 'nationality', 'international_reputation', 'skill_moves', 'weak_foot', 'work_rate_att',
               'work_rate_def', 'preferred_foot']
    return test_data_xgb, data[col_knn]


def model_process(X_train, y_train, X_test):
    # XGBoost训练过程

    model = xgb.XGBRegressor(learning_rate=0.01, n_estimators=2500, max_depth=9, min_child_weight=4, seed=0,
                             subsample=0.9, colsample_bytree=0.9, gamma=0.5, reg_alpha=0, reg_lambda=1, metrics='mae')

    model.fit(X_train, y_train)

    print(model.score(X_train, y_train))

    # 对测试集进行预测
    fit_xgb = model.predict(X_train)
    predict_xgb = model.predict(X_test)

    return fit_xgb, predict_xgb


if __name__ == '__main__':
    # #得到经过处理过的特征的训练集文件
    # deal_train('data/train.csv')
    # #得到经过处理过的特征的测试集文件
    # deal_test('data/test.csv')

    submit = pd.read_csv("data/sample_submit.csv")

    train_data_xgb, train_knn, y_train = load_trainData('data/dealed_trainInput.csv')
    # print(len(train_data_xgb.columns), len(train_knn.columns), len(y_train))
    test_data_xgb, test_knn = load_testData('data/dealed_testInput.csv')
    # print(len(test_data_xgb.columns), len(test_knn.columns))
    fit_xgb, predict_xgb = model_process(train_data_xgb, y_train, test_data_xgb)

    y_diff = y_train - fit_xgb

    model_knn = KNeighborsRegressor(n_neighbors=95)
    model_knn.fit(train_knn, y_diff)
    print(model_knn.score(train_knn, y_diff))

    predict_knn = model_knn.predict(test_knn)

    submit['y'] = predict_xgb + predict_knn
    submit.to_csv('my_xgb_Knn_prediction_2.csv', index=False)
    # cv_params = {'weights': ['distance', 'uniform']}
    # other_params = {'n_neighbors': 5, 'weights': 'uniform'}

    # cv_params = {'n_neighbors': [60, 65, 70, 75, 80, 85, 90, 95]}
    # other_params = {'n_neighbors': 95, 'weights': 'uniform'}
    #
    # model = KNeighborsRegressor(**other_params)
    # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_error', cv=5,
    #                              verbose=1, n_jobs=4)
    # optimized_GBM.fit(train_knn, y_diff)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    # =============================================================================================
    # XGBoost调试参数
    # 加载测试集，训练集
    # train_data_xgb, train_knn, y_train = load_trainData('data/dealed_trainInput.csv')
    #
    # cv_params = {'n_estimators': [1500, 2000, 2100]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'max_depth': [6, 7, 8, 9, 10], 'min_child_weight': [3, 4, 5, 6]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 2100, 'max_depth': 5, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'gamma': [0.2, 0.3, 0.4, 0.5]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 2100, 'max_depth': 9, 'min_child_weight': 4, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 2100, 'max_depth': 9, 'min_child_weight': 4, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.5, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'reg_alpha': [0, 1, 2], 'reg_lambda': [0, 1, 2]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 2100, 'max_depth': 9, 'min_child_weight': 4, 'seed': 0,
    #                 'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0.5, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'learning_rate': [0.005]}
    # other_params = {'learning_rate': 0.01, 'n_estimators': 2100, 'max_depth': 9, 'min_child_weight': 4, 'seed': 0,
    #                 'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0.5, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_error', cv=5,
    #                              verbose=1, n_jobs=4)
    # optimized_GBM.fit(train_data_xgb, y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
