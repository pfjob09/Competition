#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : LogisticRegression_Solve.py
# @Author: Huangqinjian
# @Date  : 2018/4/2
# @Desc  :
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# 加载训练数据
def loadTrainData(TrainInputFile):
    data = pd.read_csv(TrainInputFile)
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


# 加载测试数据
def loadTestData(TestInputFile):
    data = pd.read_csv(TestInputFile)
    X_testList = []
    data_len = len(data)
    for row in range(0, data_len):
        tmpList = []
        for col in range(1, data.shape[1]):
            tmpList.append(data.iloc[row][col])
        X_testList.append(tmpList)
    return X_testList


# 0.97263  38
# model = LogisticRegression()
model = LogisticRegression(tol=0.0001, solver='saga', max_iter=100, warm_start=True, C=1.0)
X_train, Y_train = loadTrainData('data/train.csv')
X_test = loadTestData('data/test.csv')
model.fit(X_train, Y_train)
print(model.score(X_train, Y_train))
ans = model.predict(X_test)
# ===========================================================================================================
result_len = len(ans)
id_list = np.arange(4000, 7550)
data_arr = []
for row in range(0, result_len):
    data_arr.append([int(id_list[row]), ans[row]])
np_data = np.array(data_arr)

# 写入文件
pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
# print(pd_data)
pd_data.to_csv('submit.csv', index=None)
# ===========================================================================================================
# 调参
# if __name__ == '__main__':
#     X_train, Y_train = loadTrainData('data/train.csv')
#     cv_params = {'C': [1.0]}
#     other_params = {'tol': 0.0001, 'solver': 'saga', 'max_iter': 100, 'warm_start': True, 'C': 1.0}
#
#     model = LogisticRegression(**other_params)
#     optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5,
#                                  verbose=1, n_jobs=4)
#     optimized_GBM.fit(X_train, Y_train)
#     evalute_result = optimized_GBM.grid_scores_
#     print('每轮迭代运行结果:{0}'.format(evalute_result))
#     print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
#     print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
