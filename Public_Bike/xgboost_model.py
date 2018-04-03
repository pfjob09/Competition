#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : bike_value.py
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
    # 原特征最佳参数
    model = xgb.XGBRegressor(learning_rate=0.07, n_estimators=200, max_depth=6, min_child_weight=6, seed=0,
                             subsample=0.8, colsample_bytree=0.8, gamma=0.3, reg_alpha=0.05, reg_lambda=3,
                             metrics='rmse')
    # 新特征最佳参数
    # model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=300, max_depth=5, min_child_weight=6, seed=0,
    #                          subsample=0.8, colsample_bytree=0.9, gamma=0.7, reg_alpha=3, reg_lambda=4,
    #                          metrics='rmse')
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


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i += 1
    outfile.close()


if __name__ == '__main__':
    # 训练模型
    # 加载测试集，训练集
    X_train, Y_train = load_trainData('data/train.csv')
    X_test = load_testData('data/test.csv')
    # 运行模型得到最终的submit.csv文件
    model_process(X_train, Y_train, X_test)
    # ==============================================================================================================
    # 加载测试集，训练集

    # X_train, Y_train = load_trainData('data/train.csv')
    # X_test = load_testData('data/test.csv')
    # model = xgb.XGBRegressor(learning_rate=0.07, n_estimators=200, max_depth=6, min_child_weight=6, seed=0,
    #                          subsample=0.8, colsample_bytree=0.8, gamma=0.3, reg_alpha=0.05, reg_lambda=3,
    #                          metrics='rmse')
    # model.fit(X_train, Y_train)
    # new_feature_train = model.apply(X_train)
    # print(new_feature_train.shape)
    # new_feature_test = model.apply(X_test)
    # print(new_feature_test.shape)
    # new_train_feature = DataFrame(new_feature_train)
    # new_test_feature = DataFrame(new_feature_test)

    # 运行模型得到最终的submit.csv文件
    # model_process(new_train_feature, Y_train, new_test_feature)
    # XGBoost调试参数
    # 加载测试集，训练集
    # X_train, Y_train = load_trainData('data/train.csv')

    # cv_params = {'n_estimators': [100, 200, 300, 400, 500]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    # cv_params = {'gamma': [0.6, 0.7, 0.8, 0.9, 1.0]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 5, 'min_child_weight': 6, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    # cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.8, 0.85, 0.9, 0.95, 1.0]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 5, 'min_child_weight': 6, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}

    # cv_params = {'reg_alpha': [0], 'reg_lambda': [1]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 5, 'min_child_weight': 6, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.9, 'gamma': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 5, 'min_child_weight': 6, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.9, 'gamma': 0.7, 'reg_alpha': 3, 'reg_lambda': 4}
    # #
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
    #                              verbose=1, n_jobs=4)
    # optimized_GBM.fit(new_train_feature, Y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    # ==============================================================================================================
    # XGBoost调试参数
    # 加载测试集，训练集
    # X_train, Y_train = load_trainData('data/train.csv')

    # cv_params = {'n_estimators': [100, 200, 300, 400, 500]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    # cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 6, 'min_child_weight': 6, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    # cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 6, 'min_child_weight': 6, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.3, 'reg_alpha': 0, 'reg_lambda': 1}

    # cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 6, 'min_child_weight': 6, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.3, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 6, 'min_child_weight': 6, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.3, 'reg_alpha': 0.05, 'reg_lambda': 3}
    #
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
    #                              verbose=1, n_jobs=4)
    # optimized_GBM.fit(X_train, Y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    # ==============================================================================================================
    # Plotting API画图，查看每棵树的结构
    # 加载测试集，训练集
    # X_train, Y_train = load_trainData('data/train.csv')
    # # 获取训练集的列名，即特征的名字
    # df = pd.read_csv('data/train.csv')
    # features = df.columns.tolist()
    # ceate_feature_map(features)
    # # XGBoost训练过程
    # model = xgb.XGBRegressor(learning_rate=0.07, n_estimators=200, max_depth=6, min_child_weight=6, seed=0,
    #                          subsample=0.8, colsample_bytree=0.8, gamma=0.3, reg_alpha=0.05, reg_lambda=3,
    #                          metrics='rmse')
    # model.fit(X_train, Y_train)
    # plot_tree(model, fmap='xgb.fmap', num_trees=5, rankdir='LR')
    # plot_importance(model)
    # plt.show()
