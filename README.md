### 这是我自己的一些比赛实战代码以及优化过程。如果大家有更好的想法，欢迎提出来！共同进步。

- SoFo_Soccer：是我接触的第一个比赛，所以可能成绩不是太好，哈哈。比赛的题目：[足球运动员身价估计](http://sofasofa.io/competition.php?id=7)。比赛调优的过程，可以参阅博客：[Scikit中的特征选择，XGboost进行回归预测，模型优化的实战](https://blog.csdn.net/sinat_35512245/article/details/79668363)
以及[XGboost数据比赛实战之调参篇(完整流程)](https://blog.csdn.net/sinat_35512245/article/details/79700029)


- Public_Bike：公共自行车使用量预测，赛题地址：[公共自行车使用量预测](http://sofasofa.io/competition.php?id=1#c4)

- SquareOrCircle：形状识别：是方还是圆，赛题地址：[形状识别：是方还是圆](http://sofasofa.io/competition.php?id=6)

- AskandQuestion：问答网站问题、回答数量预测，赛题地址：[问答网站问题、回答数量预测](http://sofasofa.io/competition.php?id=4)

- SexJudgeByName：机器读中文：根据名字判断性别，赛题地址：[机器读中文：根据名字判断性别](http://sofasofa.io/competition.php?id=3)

----------

### 一点比赛小经验：

#### 1. XGBoost的效果一般用起来都不错；

#### 2. 有时候GDBT的效果会比XGBoost好，所以当你感觉XGBoost的预测结果不是很好的时候，用一下GDBT往往会打开一个新世界的大门；

#### 3. GDBT和XGBoost两个模型的融合可以考虑一下，一般会有一定的提升。融合模型最简单的方法就是取预测结果的加权值和，并且一般预测结果好的模型对应的权重大一些，权重为0.65左右；

#### 4. 注意观察数据集，如果从特征可以看出两个明显不同的分类，可以考虑分开训练，最后再把预测的结果合并起来即可；

#### 5. 出生日期数据可以考虑转化为年龄来处理

#### 6. 身高、体重可以分段处理，或者进行归一化。
