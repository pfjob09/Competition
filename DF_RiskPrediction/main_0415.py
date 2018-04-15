import numpy as np
import pandas as pd
import xgboost as xgb

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
    data.drop(['TERMINALNO', 'TIME', 'TRIP_ID', 'Y'], axis=1, inplace=True)
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
    data.drop(['TERMINALNO', 'TIME', 'TRIP_ID'], axis=1, inplace=True)
    return data, df['TERMINALNO']


def model_process(X_train, y_train, X_test, IdList):
    model = xgb.XGBRegressor(
        learning_rate=0.001,
        n_estimators=850,
        max_depth=9,
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
