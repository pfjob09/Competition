#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : AskandQuestion.py
# @Author: Huangqinjian
# @Date  : 2018/3/31
# @Desc  :

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from xgboost import plot_tree
from pandas import DataFrame
from xgboost import plot_importance

trainFilePath = 'data/train.csv'
testFilePath = 'data/test.csv'
dealed_trainFilePath = 'data/dealed_trainInput.csv'
dealed_testFilePath = 'data/dealed_testInput.csv'


def deal_TrainInput(trainFilePath):
    data = pd.read_csv(trainFilePath)
    deal_data = data.copy(deep=True)
    deal_data['date'] = pd.to_datetime(deal_data['date'])
    deal_data['week'] = deal_data['date'].dt.dayofweek
    deal_data['month'] = deal_data['date'].dt.month
    deal_data['year'] = deal_data['date'].dt.year
    deal_data.drop(['date'], axis=1, inplace=True)

    deal_data.to_csv('data/dealed_trainInput.csv', index=False)


def deal_TestInput(testFilePath):
    data = pd.read_csv(testFilePath)
    deal_data = data.copy(deep=True)
    deal_data['date'] = pd.to_datetime(deal_data['date'])
    deal_data['week'] = deal_data['date'].dt.dayofweek
    deal_data['month'] = deal_data['date'].dt.month
    deal_data['year'] = deal_data['date'].dt.year
    deal_data.drop(['date'], axis=1, inplace=True)

    deal_data.to_csv('data/dealed_testInput.csv', index=False)


# 加载训练集
def load_trainData(trainFilePath):
    data = pd.read_csv(trainFilePath)
    X_trainList = []
    Ques_trainList = []
    Ans_trainList = []
    data_len = len(data)
    data_col = data.shape[1]
    for row in range(0, data_len):
        tmpList = []
        for col in range(3, data_col):
            tmpList.append(data.iloc[row][col])
        X_trainList.append(tmpList)
        Ques_trainList.append(data.iloc[row][1])
        Ans_trainList.append(data.iloc[row][2])
    return X_trainList, Ques_trainList, Ans_trainList


# 加载测试集
def load_testData(testFilePath):
    data = pd.read_csv(testFilePath)
    X_testList = []
    data_len = len(data)
    data_col = data.shape[1]
    for row in range(0, data_len):
        tmpList = []
        for col in range(1, data_col):
            tmpList.append(data.iloc[row][col])
        X_testList.append(tmpList)
    return X_testList


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


def model_process(X_train, Ques_trainList, Ans_trainList, X_test):
    # XGBoost训练过程
    model_ques = xgb.XGBRegressor(learning_rate=0.5, n_estimators=200, max_depth=3, min_child_weight=5, seed=0,
                                  subsample=0.95, colsample_bytree=0.001, gamma=0.05, reg_alpha=0.05, reg_lambda=0.001,
                                  metrics='rmse')

    model_ques.fit(X_train, Ques_trainList)

    # 对测试集进行预测
    ans_ques = model_ques.predict(X_test)

    model_ans = xgb.XGBRegressor(learning_rate=0.5, n_estimators=300, max_depth=3, min_child_weight=6, seed=0,
                                 subsample=0.95, colsample_bytree=0.001, gamma=0.05, reg_alpha=0.05, reg_lambda=0.001,
                                 metrics='rmse')
    model_ans.fit(X_train, Ans_trainList)

    # 对测试集进行预测
    ans_ans = model_ans.predict(X_test)

    ans_len = len(ans_ques)
    id_list = np.arange(2254, 2406)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans_ques[row], ans_ans[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'questions', 'answers'])
    # print(pd_data)
    pd_data.to_csv('submit.csv', index=None)


if __name__ == '__main__':
    # deal_TrainInput(trainFilePath)
    # deal_TestInput(testFilePath)
    # feature_plot(trainFilePath)
    X_trainList, Ques_trainList, Ans_trainList = load_trainData(dealed_trainFilePath)
    X_testList = load_testData(dealed_testFilePath)
    model_process(X_trainList, Ques_trainList, Ans_trainList, X_testList)
    # =================================================================================================
    # Plotting API画图，查看每棵树的结构
    # 加载测试集，训练集
    # X_trainList, Ques_trainList, Ans_trainList = load_trainData(dealed_trainFilePath)
    # # 获取训练集的列名，即特征的名字
    # df = pd.read_csv(dealed_trainFilePath)
    # features = df.columns[3:6].tolist()
    # print(features)
    # ceate_feature_map(features)
    # # XGBoost训练过程
    # model = xgb.XGBRegressor(learning_rate=0.5, n_estimators=200, max_depth=3, min_child_weight=5, seed=0,
    #                          subsample=0.95, colsample_bytree=0.001, gamma=0.05, reg_alpha=0.05, reg_lambda=0.001,
    #                          metrics='rmse')
    # model.fit(X_trainList, Ques_trainList)
    # plot_tree(model, fmap='xgb.fmap', num_trees=5, rankdir='LR')
    # plot_importance(model)
    # plt.show()
    # =================================================================================================
    # 利用XGBoost得到新的特征训练模型
    # 加载测试集，训练集
    # X_trainList, Ques_trainList, Ans_trainList = load_trainData(dealed_trainFilePath)
    # X_testList = load_testData(dealed_testFilePath)
    # model_ques = xgb.XGBRegressor(learning_rate=0.5, n_estimators=200, max_depth=3, min_child_weight=5, seed=0,
    #                               subsample=0.95, colsample_bytree=0.001, gamma=0.05, reg_alpha=0.05, reg_lambda=0.001,
    #                               metrics='rmse')
    # model_ques.fit(X_trainList, Ques_trainList)
    # # 得到新的训练集的特征
    # new_X_train = model_ques.apply(X_trainList)
    # print(new_X_train.shape)
    # # 得到新的测试集的特征
    # new_X_test = model_ques.apply(X_testList)
    # print(new_X_test.shape)
    # # 得到的新特征DataFrame化
    # new_train_feature = DataFrame(new_X_train)
    # new_test_feature = DataFrame(new_X_test)
    # # 运行模型得到最终的submit.csv文件
    # model_process(new_train_feature, Ques_trainList, Ans_trainList, new_test_feature)
