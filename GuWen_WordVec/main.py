import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec

train = pd.read_csv('data/train.txt')
test = pd.read_csv('data/test.txt')
submit = pd.read_csv('data/sample_submit.csv')

total = len(train) + len(test)
n_train = len(train)

labeled_texts = []

texts = list(train['text']) + list(test['text'])

ndims = 100
model = Word2Vec(sentences=texts, size=ndims)

vecs = np.zeros([total, ndims])
for i, sentence in enumerate(texts):
    counts, row = 0, 0
    for char in sentence:
        try:
            if char != ' ':
                row += model.wv[char]
                counts += 1
        except:
            pass
    if counts == 0:
        print(sentence)
    vecs[i, :] = row / counts

clf = xgb.XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.01,
    n_estimators=1000,
    max_depth=4,
    min_child_weight=5,
    seed=0,
    subsample=0.8,
    colsample_bytree=0.3,
    gamma=0.5,
    reg_alpha=3,
    reg_lambda=1,
    metrics='logloss')
clf.fit(vecs[:n_train], train['y'])
print(clf.score(vecs[:n_train], train['y']))
submit['y'] = clf.predict_proba(vecs[n_train:])[:, 1]
submit.to_csv('my_prediction.csv', index=False)



def model_cv(X_train, Y_train):
    # cv_params = {
    #     'max_depth': range(1, 7, 1),
    #     # 'min_samples_split': range(4, 9, 1),
    #     # 'min_samples_leaf ': [0.5, 0.6, 0.7, 0.8, 0.9],
    # }
    # model = DecisionTreeClassifier(max_depth=5, random_state=100)

    cv_params = {
        'n_estimators': range(1000, 1050, 25),
        # 'max_depth': range(3, 8, 1),
        # 'min_child_weight': range(3, 8, 1),
        # 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
        # 'colsample_bytree': [0.2, 0.3, 0.4],
        # 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        # 'reg_alpha': [2, 3, 4, 5, 6],
        # 'reg_lambda': [2, 3, 4, 5, 6, 7],
        # 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1]
    }
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=5,
        seed=0,
        subsample=0.8,
        colsample_bytree=0.3,
        gamma=0.5,
        reg_alpha=3,
        reg_lambda=1,
        metrics='logloss')

    optimized_GBM = GridSearchCV(
        estimator=model,
        param_grid=cv_params,
        scoring='neg_log_loss',
        cv=5,
        verbose=1,
        n_jobs=4)

    optimized_GBM.fit(X_train, Y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


# if __name__ == "__main__":
#     model_cv(vecs[:n_train], train['y'])
