#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : gbdt_model_learning_curve.py
# @Author: Huangqinjian
# @Date  : 2018/3/28
# @Desc  :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

from matplotlib import markers

markers.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
markers.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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
    # GBDT训练过程
    model = GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=200, subsample=0.6,
                                      min_samples_split=3,
                                      min_samples_leaf=5, max_depth=6, warm_start=True)

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


# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


if __name__ == '__main__':
    # 加载测试集，训练集
    X_train, Y_train = load_trainData('data/train.csv')
    model = GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=200, subsample=0.6,
                                      min_samples_split=3,
                                      min_samples_leaf=5, max_depth=6, warm_start=True)
    plot_learning_curve(model, u"学习曲线", X_train, Y_train, cv=5)
    # ====================================================================================================================
    # X_train, Y_train = load_trainData('data/train.csv')
    # X_test = load_testData('data/test.csv')
    # model_process(X_train, Y_train, X_test)
    # ====================================================================================================================
    # GBDT调试参数
    # cv_params = {'n_estimators': [100, 200, 300, 400, 500]}
    # other_params = {'loss': 'ls', 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.8,
    #                 'min_samples_split': 2,
    #                 'min_samples_leaf': 1, 'max_depth': 5, 'warm_start': True}

    # cv_params = {'max_depth': [4, 5, 6, 7, 8]}
    # other_params = {'loss': 'ls', 'learning_rate': 0.05, 'n_estimators': 200, 'subsample': 0.6,
    #                 'min_samples_split': 2,
    #                 'min_samples_leaf': 4, 'max_depth': 6, 'warm_start': True}

    # cv_params = {'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    # other_params = {'loss': 'ls', 'learning_rate': 0.05, 'n_estimators': 200, 'subsample': 0.6,
    #                 'min_samples_split': 2,
    #                 'min_samples_leaf': 4, 'max_depth': 6, 'warm_start': True}

    # cv_params = {'min_samples_split': [2, 3, 4, 5, 6], 'min_samples_leaf': [2, 3, 4, 5, 6]}
    # other_params = {'loss': 'ls', 'learning_rate': 0.05, 'n_estimators': 200, 'subsample': 0.6,
    #                 'min_samples_split': 3,
    #                 'min_samples_leaf': 5, 'max_depth': 6, 'warm_start': True}

    # cv_params = {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 200, 300, 400, 500]}
    # other_params = {'loss': 'ls', 'learning_rate': 0.05, 'n_estimators': 200, 'subsample': 0.6,
    #                 'min_samples_split': 2,
    #                 'min_samples_leaf': 4, 'max_depth': 6, 'warm_start': True}

    # model = GradientBoostingRegressor(**other_params)
    # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
    #                              verbose=1, n_jobs=4)
    # optimized_GBM.fit(X_train, Y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
