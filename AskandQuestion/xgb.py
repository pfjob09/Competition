#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : AskandQuestion.py
# @Author: Huangqinjian
# @Date  : 2018/3/31
# @Desc  :

import numpy as np
import pandas as pd
import xgboost as xgb
from chinese_calendar import is_workday, is_holiday
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from xgboost import plot_tree
from pandas import DataFrame
from xgboost import plot_importance
import operator
import win_unicode_console
win_unicode_console.enable()

trainFilePath = 'data/train.csv'
testFilePath = 'data/test.csv'
dealed_trainFilePath = 'data/dealed_trainInput.csv'
dealed_testFilePath = 'data/dealed_testInput.csv'
data_submitPath = 'data/sample_submit.csv'


def deal_TrainInput(trainFilePath):
    data = pd.read_csv(trainFilePath)
    deal_data = data.copy(deep=True)
    deal_data['date'] = pd.to_datetime(deal_data['date'])
    deal_data['week'] = deal_data['date'].dt.dayofweek
    deal_data['month'] = deal_data['date'].dt.month
    deal_data['year'] = deal_data['date'].dt.year
    deal_data['is_workday'] = deal_data['date'].apply(
        lambda x: 1 if is_workday(pd.to_datetime(x)) else 0)
    deal_data['is_holiday'] = deal_data['date'].apply(
        lambda x: 1 if is_holiday(pd.to_datetime(x)) else 0)
    deal_data.drop(['date'], axis=1, inplace=True)

    deal_data.to_csv('data/dealed_trainInput.csv', index=False)


def deal_TestInput(testFilePath):
    data = pd.read_csv(testFilePath)
    deal_data = data.copy(deep=True)
    deal_data['date'] = pd.to_datetime(deal_data['date'])
    deal_data['week'] = deal_data['date'].dt.dayofweek
    deal_data['month'] = deal_data['date'].dt.month
    deal_data['year'] = deal_data['date'].dt.year
    deal_data['is_workday'] = deal_data['date'].apply(
        lambda x: 1 if is_workday(pd.to_datetime(x)) else 0)
    deal_data['is_holiday'] = deal_data['date'].apply(
        lambda x: 1 if is_holiday(pd.to_datetime(x)) else 0)
    deal_data.drop(['date'], axis=1, inplace=True)

    deal_data.to_csv('data/dealed_testInput.csv', index=False)


def feature_plot(FilePath):
    data_train = pd.read_csv(FilePath)
    # 将数据类型转换为日期类型
    data_train['date'] = pd.to_datetime(data_train['date'])
    # 将date设置为index
    df = data_train.set_index('date')
    # ===========================================================================================================
    # 获取某年的数据
    # df_some_year = df['2010']
    # x_trick = np.arange(len(df_some_year))
    # plt.plot(x_trick, df_some_year['questions'], label='questions')
    # plt.plot(x_trick, df_some_year['answers'], label='answers')
    # plt.title('2010的问题数，回答数变化趋势')
    # plt.legend(['questions', 'answers'], loc=2)
    # plt.show()
    # ===========================================================================================================
    # 获取某月的数据
    # df_some_month = df['2013-11']
    # x_trick = np.arange(len(df_some_month))
    # plt.plot(x_trick, df_some_month['questions'], label='questions')
    # plt.plot(x_trick, df_some_month['answers'], label='answers')
    # plt.title('2013-11的问题数，回答数变化趋势')
    # plt.legend(['questions', 'answers'], loc=2)
    # plt.show()
    # ===========================================================================================================
    # 获取2015-12-31之前的数据
    # df_after = df.truncate(after='2015-12-31')
    # x_trick = np.arange(len(df_after))
    # plt.plot(x_trick, df_after['questions'], label='questions')
    # plt.plot(x_trick, df_after['answers'], label='answers')
    # plt.title('2015-12-31之前的问题数，回答数变化趋势')
    # plt.legend(['questions', 'answers'], loc=2)
    # plt.show()
    # ===========================================================================================================
    # 获取2016-01-01以后的数据
    # df_before = df.truncate(before='2016-01-01')
    # x_trick = np.arange(len(df_before))
    # plt.plot(x_trick, df_before['questions'], label='questions')
    # plt.plot(x_trick, df_before['answers'], label='answers')
    # plt.title('2016-01-01以后的问题数，回答数变化趋势')
    # plt.legend(['questions', 'answers'], loc=2)
    # plt.show()
    # ===========================================================================================================
    # 获取两段时间之间的数据
    # df_between = df.truncate(before='2016-05-02', after='2016-08-15')
    # x_trick = np.arange(len(df_between))
    # plt.plot(x_trick, df_between['questions'], label='questions')
    # plt.plot(x_trick, df_between['answers'], label='answers')
    # plt.title('2016-05-02到2016-08-15之间的问题数，回答数变化趋势')
    # plt.legend(['questions', 'answers'], loc=2)
    # plt.show()
    # ===========================================================================================================
    # 获取两段时间之间的问题数，回答数差值
    # df_between = df.truncate(before='2016-05-02', after='2016-08-15')
    # x_trick = np.arange(len(df_between))
    # plt.plot(x_trick, df_between['answers'] - df_between['questions'], label='差值')
    # plt.title('2016-05-02到2016-08-15之间的问题数，回答数差值变化趋势')
    # plt.legend(['差值'], loc=2)
    # plt.show()
    # ===========================================================================================================
    # 按年统计并显示
    # df_year = df.resample('AS').sum().to_period('A')
    # x_trick = np.arange(len(df_year))
    # plt.plot(x_trick, df_year['questions'], label='questions')
    # plt.plot(x_trick, df_year['answers'], label='answers')
    # plt.title('每年问题数，回答数变化趋势')
    # plt.legend(['questions', 'answers'], loc=2)
    # plt.show()
    # ===========================================================================================================
    # 按季度统计并显示
    # df_quarter = df.resample('Q').sum().to_period('Q')
    # x_trick = np.arange(len(df_quarter))
    # plt.plot(x_trick, df_quarter['questions'], label='questions')
    # plt.plot(x_trick, df_quarter['answers'], label='answers')
    # plt.title('每季度问题数，回答数变化趋势')
    # plt.legend(['questions', 'answers'], loc=2)
    # plt.show()
    # ===========================================================================================================
    # 按月度统计并显示
    # df_month = df.resample('M').sum().to_period('M')
    # x_trick = np.arange(len(df_month))
    # plt.plot(x_trick, df_month['questions'], label='questions')
    # plt.plot(x_trick, df_month['answers'], label='answers')
    # plt.title('每月问题数，回答数变化趋势')
    # plt.legend(['questions', 'answers'], loc=2)
    # plt.show()
    # ===========================================================================================================
    # 按周统计并显示
    # df_week = df_week = df.resample('W').sum().to_period('W')
    # x_trick = np.arange(len(df_week))
    # plt.plot(x_trick, df_week['questions'], label='questions')
    # plt.plot(x_trick, df_week['answers'], label='answers')
    # plt.title('每周问题数，回答数变化趋势')
    # plt.legend(['questions', 'answers'], loc=2)
    # plt.show()


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i += 1
    outfile.close()


def get_feature_importance(X_train, Y_train):
    features = X_train.columns.tolist()
    ceate_feature_map(features)
    # params = {
    #     'learning_rate': 0.1,
    #     'n_estimators': 1650,
    #     'max_depth': 2,
    #     'min_child_weight': 5,
    #     'seed': 0,
    #     'subsample': 0.6,
    #     'colsample_bytree': 0.8,
    #     'gamma': 0.1,
    #     'reg_alpha': 1,
    #     'reg_lambda': 4,
    #     'metrics': 'rmse'
    # }

    params = {
        'learning_rate': 0.1,
        'n_estimators': 1350,
        'max_depth': 2,
        'min_child_weight': 6,
        'seed': 0,
        'subsample': 0.65,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 5,
        'reg_lambda': 3,
        'metrics': 'rmse'
    }

    xgbtrain = xgb.DMatrix(X_train, label=Y_train)
    watchlist = [(xgbtrain, 'train')]
    bst = xgb.train(
        params,
        xgbtrain,
        num_boost_round=3500,
        evals=watchlist,
        early_stopping_rounds=50)

    importance = bst.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.plot(
        kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.show()


def xgb_cv(X_train, Y_train):
    cv_params = {
        # 'n_estimators': range(1200, 1500, 50),
        # 'max_depth': range(1, 6, 1),
        # 'min_child_weight': range(5, 9, 1),
        # 'subsample':
        # [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        # 'colsample_bytree': [0.8],
        # 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        # 'reg_alpha': [4, 5, 6, 7, 8, 9],
        # 'reg_lambda': [3],
        'learning_rate': [0.01, 0.05, 0.1, 0.15]
    }
    model = xgb.XGBRegressor(
        learning_rate=0.1,
        n_estimators=1350,
        max_depth=2,
        min_child_weight=6,
        seed=0,
        subsample=0.65,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=5,
        reg_lambda=3,
        metrics='rmse')
    optimized_GBM = GridSearchCV(
        estimator=model,
        param_grid=cv_params,
        scoring='neg_mean_absolute_error',
        cv=5,
        verbose=1,
        n_jobs=4)
    optimized_GBM.fit(X_train, Y_train)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


def model_process(X_train, Ques_trainList, Ans_trainList, X_test):
    # XGBoost训练过程
    model_ques = xgb.XGBRegressor(
        learning_rate=0.1,
        n_estimators=1650,
        max_depth=2,
        min_child_weight=5,
        seed=0,
        subsample=0.6,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=1,
        reg_lambda=4,
        metrics='rmse')

    model_ques.fit(X_train, Ques_trainList)

    # 对测试集进行预测
    ans_ques = model_ques.predict(X_test)

    model_ans = xgb.XGBRegressor(
        learning_rate=0.1,
        n_estimators=1350,
        max_depth=2,
        min_child_weight=6,
        seed=0,
        subsample=0.65,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=5,
        reg_lambda=3,
        metrics='rmse')
    model_ans.fit(X_train, Ans_trainList)

    # 对测试集进行预测
    ans_ans = model_ans.predict(X_test)

    submit_data = pd.read_csv(data_submitPath)
    submit_data['questions'] = ans_ques
    submit_data['answers'] = ans_ans
    submit_data.to_csv('result.csv', index=None)


if __name__ == '__main__':
    # deal_TrainInput(trainFilePath)
    # deal_TestInput(testFilePath)
    # feature_plot(trainFilePath)
    train = pd.read_csv(dealed_trainFilePath)
    X_trainList = train.drop(['id', 'questions', 'answers'], axis=1)
    Ques_trainList = train['questions']
    Ans_trainList = train['answers']

    test = pd.read_csv(dealed_testFilePath)

    X_testList = test.drop('id', axis=1)

    # get_feature_importance(X_trainList, Ques_trainList)
    # get_feature_importance(X_trainList, Ans_trainList)
    # xgb_cv(X_trainList, Ques_trainList)
    # xgb_cv(X_trainList, Ans_trainList)

    model_process(X_trainList, Ques_trainList, Ans_trainList, X_testList)
