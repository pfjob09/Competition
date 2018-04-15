import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import win_unicode_console
win_unicode_console.enable()

path_train = "data/dm/train.csv"  # 训练文件

data_train = pd.read_csv(path_train)

# sns.heatmap(data_train.corr(), annot=True)

# plt.show()

#=========================================================================================


def time_convert(timestamp):
    #转换成localtime
    time_local = time.localtime(timestamp)
    #转换成新的时间格式(2016-05-05 20:28:54)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return dt


# data_train['TIME'] = data_train['TIME'].apply(lambda x: time_convert(x))

# print(data_train['TIME'].head(60))

#=========================================================================================

average_speed = data_train['SPEED'].groupby(data_train['TERMINALNO']).mean()

average_height = data_train['HEIGHT'].groupby(data_train['TERMINALNO']).mean()

callstate = data_train[data_train['TERMINALNO'] == 100][
    'CALLSTATE'].value_counts()

# data = pd.DataFrame()

# data['height'] = average_height
# data['speed'] = average_speed
# data['y'] = data_train['Y'].groupby(data_train['TERMINALNO']).mean()

# print(data)

print(callstate)
