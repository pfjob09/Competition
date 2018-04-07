#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : xgb_finally.py
# @Author: Huangqinjian
# @Date  : 2018/4/6
# @Desc  :

import pandas as pd
import xgboost as xgb
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


#  出生日期转化为年龄，对结果提升很大
def birthdate_age(birthdate):
    year = int(birthdate.split('/')[2])
    if year <= 18:
        return 18 - year
    else:
        return 118 - year


# 对体重进行分段处理
def weight_map(weight):
    if (weight >= 68) and (weight <= 80):
        return 1
    elif (weight >= 62) and (weight < 68) or (weight > 80) and (weight <= 86):
        return 2
    else:
        return 3


# 对身高进行分段处理
def height_map(height):
    if (height >= 175) and (height <= 188):
        return 1
    elif (height >= 165) and (height < 175) or (height > 188) and (height <= 194):
        return 2
    else:
        return 3


# 对国籍进行分段处理，本题中作用不大
def nationality_map(nationality):
    if (nationality >= 50) and (nationality <= 65) or (nationality >= 98) and (nationality <= 115) or (
                nationality >= 147) and (nationality <= 163):
        return 1
    elif (nationality >= 1) and (nationality <= 17) or (nationality >= 82) and (nationality < 98) or (
                nationality >= 131) and (nationality < 147):
        return 2
    else:
        return 3


# 处理训练集的数据，进行特征处理，输出经过处理过的特征的训练集文件
def deal_train(trainFilePath):
    data = pd.read_csv(trainFilePath)
    # 守门员
    # 此处要进行copy，不然会报错
    train_goalkeeper_data = data[~data['gk'].isnull()].copy()
    # 其他人员
    # 此处要进行copy，不然会报错
    train_soccer_data = data[data['gk'].isnull()].copy()

    # 离散型特征映射
    mapper = {'work_rate_att': {'Low': 1, 'Medium': 2, 'High': 3}, 'work_rate_def': {'Low': 1, 'Medium': 2, 'High': 3}}

    # 映射转换
    for col, mapItem in mapper.items():
        train_goalkeeper_data.loc[:, col] = train_goalkeeper_data[col].map(mapItem)
        train_soccer_data.loc[:, col] = train_soccer_data[col].map(mapItem)

    # 丢弃的列
    goalkeeper_dropList = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb']

    # 根据出生日期计算年龄
    train_goalkeeper_data['birth_date'] = train_goalkeeper_data['birth_date'].apply(birthdate_age)
    train_soccer_data['birth_date'] = train_soccer_data['birth_date'].apply(birthdate_age)

    # 对体重进行分段处理
    train_goalkeeper_data['weight_kg'] = train_goalkeeper_data['weight_kg'].apply(weight_map)
    train_soccer_data['weight_kg'] = train_soccer_data['weight_kg'].apply(weight_map)

    # 对身高进行分段处理
    train_goalkeeper_data['height_cm'] = train_goalkeeper_data['height_cm'].apply(height_map)
    train_soccer_data['height_cm'] = train_soccer_data['height_cm'].apply(height_map)

    # 对国籍进行分段处理，本题中作用不大
    # train_goalkeeper_data['nationality'] = train_goalkeeper_data['nationality'].apply(nationality_map)
    # train_soccer_data['nationality'] = train_soccer_data['nationality'].apply(nationality_map)

    train_goalkeeper_data.drop(goalkeeper_dropList, axis=1, inplace=True)
    train_soccer_data.drop(['gk'], axis=1, inplace=True)

    # 输出经过处理过的特征的训练集文件
    train_goalkeeper_data.to_csv('data/dealed_goalkeeper_trainInput.csv', index=False)
    train_soccer_data.to_csv('data/dealed_soccer_trainInput.csv', index=False)


# 处理测试集的数据，进行特征处理，输出经过处理过的特征的测试集文件
def deal_test(testFilePath):
    data = pd.read_csv(testFilePath)
    # 守门员
    test_goalkeeper_data = data[~data['gk'].isnull()].copy()
    # 其他人员
    test_soccer_data = data[data['gk'].isnull()].copy()

    # 离散型特征映射
    mapper = {'work_rate_att': {'Low': 1, 'Medium': 2, 'High': 3}, 'work_rate_def': {'Low': 1, 'Medium': 2, 'High': 3}}

    # 映射转换
    for col, mapItem in mapper.items():
        test_goalkeeper_data.loc[:, col] = test_goalkeeper_data[col].map(mapItem)
        test_soccer_data.loc[:, col] = test_soccer_data[col].map(mapItem)

    # 丢弃的列
    goalkeeper_dropList = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb']

    # 根据出生日期计算年龄
    test_goalkeeper_data['birth_date'] = test_goalkeeper_data['birth_date'].apply(birthdate_age)
    test_soccer_data['birth_date'] = test_soccer_data['birth_date'].apply(birthdate_age)

    # 对体重进行分段处理
    test_goalkeeper_data['weight_kg'] = test_goalkeeper_data['weight_kg'].apply(weight_map)
    test_soccer_data['weight_kg'] = test_soccer_data['weight_kg'].apply(weight_map)

    # 对身高进行分段处理
    test_goalkeeper_data['height_cm'] = test_goalkeeper_data['height_cm'].apply(height_map)
    test_soccer_data['height_cm'] = test_soccer_data['height_cm'].apply(height_map)

    test_goalkeeper_data.drop(goalkeeper_dropList, axis=1, inplace=True)
    test_soccer_data.drop(['gk'], axis=1, inplace=True)

    # 输出经过处理过的特征的训练集文件
    test_goalkeeper_data.to_csv('data/dealed_goalkeeper_testInput.csv', index=False)
    test_soccer_data.to_csv('data/dealed_soccer_testInput.csv', index=False)


# 加载训练集
def load_trainData(soccer_trainFilePath, goalkeeper_trainFilePath):
    goalkeeper_data = pd.read_csv(goalkeeper_trainFilePath)
    soccer_data = pd.read_csv(soccer_trainFilePath)
    train_goalkeeper_data = goalkeeper_data.copy()
    train_soccer_data = soccer_data.copy()

    train_goalkeeper_data.drop(['id', 'y'], axis=1, inplace=True)
    train_soccer_data.drop(['id', 'y'], axis=1, inplace=True)

    return train_goalkeeper_data, train_soccer_data, goalkeeper_data['y'], soccer_data['y']


# 加载测试集
def load_testData(soccer_testFilePath, goalkeeper_testFilePath):
    goalkeeper_data = pd.read_csv(goalkeeper_testFilePath)
    soccer_data = pd.read_csv(soccer_testFilePath)
    test_goalkeeper_data = goalkeeper_data.copy()
    test_soccer_data = soccer_data.copy()

    test_goalkeeper_data.drop(['id'], axis=1, inplace=True)
    test_soccer_data.drop(['id'], axis=1, inplace=True)

    return test_goalkeeper_data, test_soccer_data, goalkeeper_data['id'], soccer_data['id']


def xgb_cv(X_train, Y_train):
    cv_params = {
        # 'n_estimators': range(50, 120, 10),
        # 'max_depth': range(7, 10, 1),
        # 'min_child_weight': range(2, 7, 1),
        # 'subsample': [0.5, 0.55, 0.6, 0.65],
        # 'colsample_bytree': [0.8, 0.9, 0.95],
        # 'gamma': [0.1, 0.2, 0.3, 0.4, 0.45],
        # 'reg_alpha': [0, 1, 2, 3, 4, 5],
        # 'reg_lambda': [0, 1, 2, 3, 4, 5],
        # 'learning_rate': [0.05, 0.1, 0.15]

        # 'n_estimators': [1200],
        # 'max_depth': range(6, 10, 1),
        # 'min_child_weight': range(2, 7, 1),
        # 'subsample': [0.75, 0.85],
        # 'colsample_bytree': [0.8, 0.9, 0.95],
        # 'gamma': [0.3, 0.4, 0.5, 0.6],
        # 'reg_alpha': [2, 3, 4, 5],
        # 'reg_lambda': [2, 3, 4, 5],
        # 'learning_rate': [0.01, 0.05, 0.1]
    }
    # -18.27977
    model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=110, max_depth=9, min_child_weight=4, seed=0,
                             subsample=0.6, colsample_bytree=0.95, gamma=0.7, reg_alpha=3, reg_lambda=3,
                             metrics='mae')
    # -24.890
    # model = xgb.XGBRegressor(learning_rate=0.05, n_estimators=1200, max_depth=9, min_child_weight=6, seed=0,
    #                          subsample=0.8, colsample_bytree=0.9, gamma=0.4, reg_alpha=5, reg_lambda=3,
    #                          metrics='mae')
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_error', cv=5,
                                 verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, Y_train)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


def model_process(X_train, y_train, X_test, idList):
    # XGBoost训练过程
    model_goalkeeper = xgb.XGBRegressor(learning_rate=0.1, n_estimators=110, max_depth=9, min_child_weight=4, seed=0,
                                        subsample=0.6, colsample_bytree=0.95, gamma=0.7, reg_alpha=3, reg_lambda=3,
                                        metrics='mae')

    model_goalkeeper.fit(X_train, y_train)
    #  print(model.score(X_train, y_train))

    # 对测试集进行预测
    predict_goalkeeper = model_goalkeeper.predict(X_test)

    submit = DataFrame(idList)
    submit['y'] = predict_goalkeeper
    submit.to_csv('goalkeeper.csv', index=False)
    """
    model_soccer = xgb.XGBRegressor(learning_rate=0.05, n_estimators=1200, max_depth=9, min_child_weight=6, seed=0,
                                    subsample=0.8, colsample_bytree=0.9, gamma=0.4, reg_alpha=5, reg_lambda=3,
                                    metrics='mae')

    model_soccer.fit(X_train, y_train)
    predict_soccer = model_soccer.predict(X_test)
    submit = DataFrame(idList)
    submit['y'] = predict_soccer
    submit.to_csv('soccer.csv', index=False)
    """


def plot_feature():
    data = pd.read_csv('data/dealed_soccer_trainInput.csv')
    # height_cm, weight_kg
    # data['height_cm'].plot(kind='hist')
    # data['weight_kg'].plot(kind='hist')
    # data['club'].plot(kind='hist')
    # data['league'].plot(kind='hist')
    # data['nationality'].plot(kind='hist')
    plt.show()


if __name__ == '__main__':
    # plot_feature()
    # #得到经过处理过的特征的训练集文件
    # deal_train('data/train.csv')
    # #得到经过处理过的特征的测试集文件
    # deal_test('data/test.csv')
    # =============================================================================================
    # XGBoost调试参数
    # 加载测试集，训练集
    # train_goalkeeper_data, train_soccer_data, goalkeeper_train_y, soccer_train_y = load_trainData(
    #     'data/dealed_soccer_trainInput.csv',
    #     'data/dealed_goalkeeper_trainInput.csv')
    # print(train_goalkeeper_data.tail())

    # xgb_cv(train_goalkeeper_data, goalkeeper_train_y)
    # xgb_cv(train_soccer_data, soccer_train_y)

    # test_goalkeeper_data, test_soccer_data, goalkeeper_id, soccer_id = load_testData(
    #     'data/dealed_soccer_testInput.csv',
    #     'data/dealed_goalkeeper_testInput.csv')
    # print(goalkeeper_id, soccer_id)
    # =============================================================================================
    # model_process(train_goalkeeper_data, goalkeeper_train_y, test_goalkeeper_data, goalkeeper_id)
    # model_process(train_soccer_data, soccer_train_y, test_soccer_data, soccer_id)
    # =============================================================================================
    # 合并结果
    data_sample = pd.read_csv('data/sample_submit.csv')
    data_soccer = pd.read_csv('soccer.csv')
    data_goalkeeper = pd.read_csv('goalkeeper.csv')
    merge_goalkeeper = pd.merge(data_sample, data_goalkeeper, on='id', how='left')
    merge_soccer = pd.merge(merge_goalkeeper, data_soccer, on='id', how='left')
    merge_soccer.fillna(0, inplace=True)
    merge_soccer['y'] = merge_soccer['y'] + merge_soccer['y_y']
    merge_soccer.to_csv('submit.csv', index=False)
