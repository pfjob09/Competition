import numpy as np
import pandas as pd
import time
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def speed_map(speed):
    if speed <= 4.5:
        return 1
    elif (speed > 4.5) and (speed <= 15):
        return 2
    elif (speed > 15) and (speed <= 31.5):
        return 3
    else:
        return 4


def height_map(hegiht):
    if hegiht <= 91.5:
        return 1
    elif (hegiht > 91.5) and (hegiht <= 451):
        return 2
    elif (hegiht > 451) and (hegiht <= 1570):
        return 3
    else:
        return 4


def speed_height(speed, height):
    if height <= 451:
        return 1
    elif height > 451 and height <= 1570 and speed > 15:
        return 2
    elif height > 451 and height <= 1570 and speed > 31.5:
        return 3
    elif height > 1570 and speed > 31.5:
        return 4
    else:
        return 5


def longtitude_map(longti):
    if (longti >= 118) and (longti <= 123):
        return 1
    elif (longti >= 105) and (longti <= 118):
        return 2
    elif (longti >= 101) and (longti < 105) or (longti > 123) and (longti <=
                                                                   127):
        return 3
    else:
        return 4


def direction_map(direction):
    if direction >= 0 and direction < 90:
        return 1
    elif direction >= 90 and direction < 180:
        return 2
    elif direction >= 180 and direction < 270:
        return 3
    elif direction >= 270 and direction < 360:
        return 4
    else:
        return 5


def speed_direction(speed, direction):
    if direction < 180 and speed < 31.5:
        return 1
    elif direction < 180 and speed > 31.5:
        return 2
    elif direction > 180 and direction < 360 and speed < 31.5:
        return 3
    elif direction > 180 and direction < 360 and speed > 31.5:
        return 4
    else:
        return 5


def time_convert(timestamp, type):
    #转换成localtime
    time_local = time.localtime(timestamp)
    if type == 'hour':
        #转换成新的时间格式(2016-05-05 20:28:54)
        # dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
        dt = time.strftime("%H", time_local)
    else:
        dt = time.strftime("%w", time_local)
    return dt


# 加载训练集
def load_trainData(trainFilePath):
    df = pd.read_csv(trainFilePath)
    data = df.copy()
    data['DIRECTION'] = data['DIRECTION'].apply(direction_map)
    data['SPEED'] = data['SPEED'].apply(speed_map)
    data['HEIGHT'] = data['HEIGHT'].apply(height_map)
    data['LONGITUDE'] = data['LONGITUDE'].apply(longtitude_map)
    data['SPEED_HEIGHT'] = data.apply(
        lambda col: speed_height(col['SPEED'], col['HEIGHT']), axis=1)
    data['SPEED_DIRECTION'] = data.apply(
        lambda col: speed_direction(col['SPEED'], col['DIRECTION']), axis=1)
    data['Hour'] = data['TIME'].apply(lambda x: time_convert(x, 'hour'))
    data['Week'] = data['TIME'].apply(lambda x: time_convert(x, 'week'))
    data[['Hour', 'Week']] = data[['Hour', 'Week']].apply(pd.to_numeric)
    data['isNight'] = data['Hour'].apply(
        lambda x: 0 if (x > 5 and x < 19) else 1)
    data['isWeekend'] = data['Week'].apply(
        lambda x: 0 if (x > 0 and x < 6) else 1)
    data.drop(
        ['TERMINALNO', 'TIME', 'TRIP_ID', 'Hour', 'Week', 'Y'],
        axis=1,
        inplace=True)
    return data, df['Y']


# 加载测试集
def load_testData(testFilePath):
    df = pd.read_csv(testFilePath)
    data = df.copy()
    data['DIRECTION'] = data['DIRECTION'].apply(direction_map)
    data['SPEED'] = data['SPEED'].apply(speed_map)
    data['HEIGHT'] = data['HEIGHT'].apply(height_map)
    data['LONGITUDE'] = data['LONGITUDE'].apply(longtitude_map)
    data['SPEED_HEIGHT'] = data.apply(
        lambda col: speed_height(col['SPEED'], col['HEIGHT']), axis=1)
    data['SPEED_DIRECTION'] = data.apply(
        lambda col: speed_direction(col['SPEED'], col['DIRECTION']), axis=1)
    data['Hour'] = data['TIME'].apply(lambda x: time_convert(x, 'hour'))
    data['Week'] = data['TIME'].apply(lambda x: time_convert(x, 'week'))
    data[['Hour', 'Week']] = data[['Hour', 'Week']].apply(pd.to_numeric)
    data['isNight'] = data['Hour'].apply(
        lambda x: 0 if (x > 5 and x < 19) else 1)
    data['isWeekend'] = data['Week'].apply(
        lambda x: 0 if (x > 0 and x < 6) else 1)
    data.drop(
        ['TERMINALNO', 'TIME', 'TRIP_ID', 'Hour', 'Week'],
        axis=1,
        inplace=True)
    return data, df['TERMINALNO']


def xgb_cv(X_train, Y_train):
    cv_params = {
        # 'n_estimators': range(1350, 2200, 50),
        # 'max_depth': range(4, 9, 1),
        # 'min_child_weight': range(4, 9, 1),
        # 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
        # 'colsample_bytree': [0.1, 0.2, 0.3, 0.4],
        # 'gamma': [0.2, 0.3, 0.4, 0.5, 0.6],
        # 'reg_alpha': [2, 3, 4, 5, 6],
        # 'reg_lambda': [2, 3, 4, 5, 6, 7],
        # 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1]
    }
    model = xgb.XGBRegressor(
        learning_rate=0.001,
        n_estimators=1500,
        max_depth=6,
        min_child_weight=5,
        seed=0,
        subsample=0.8,
        colsample_bytree=0.3,
        gamma=0.1,
        reg_alpha=3,
        reg_lambda=1,
        metrics='auc')

    optimized_GBM = GridSearchCV(
        estimator=model,
        param_grid=cv_params,
        scoring='neg_mean_absolute_error',
        cv=5,
        verbose=1,
        n_jobs=4)

    optimized_GBM.fit(X_train, Y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


def model_process(X_train, y_train, X_test, IdList):
    model = xgb.XGBRegressor(
        learning_rate=0.001,
        n_estimators=1800,
        max_depth=6,
        min_child_weight=5,
        seed=0,
        subsample=0.8,
        colsample_bytree=0.3,
        gamma=0.1,
        reg_alpha=3,
        reg_lambda=1,
        metrics='auc')
    model.fit(X_train, y_train)
    ans = model.predict(X_test)

    ans_len = len(ans)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(IdList[row]), ans[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['Id', 'Pred'])
    pd_data.to_csv('model/submit.csv', index=None)

    data_submit = pd.read_csv('model/submit.csv')
    pred_average = data_submit['Pred'].groupby(
        data_submit['Id']).mean().tolist()
    df = data_submit.drop_duplicates(subset='Id')
    id_list = df['Id'].tolist()

    ans_len = len(pred_average)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), 0 - pred_average[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['Id', 'Pred'])
    pd_data.to_csv('model/submit.csv', index=None)


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    X_train, Y_train = load_trainData(path_train)
    X_test, IdList = load_testData(path_test)
    model_process(X_train, Y_train, X_test, IdList)
    # xgb_cv(X_train, Y_train)
