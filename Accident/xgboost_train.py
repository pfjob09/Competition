# coding=utf-8
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def my_custom_loss_func(ground_truth, predictions):
    return average_precision_score(ground_truth, predictions)


def xgb_cv(X_train, Y_train):
    score = make_scorer(my_custom_loss_func, greater_is_better=True)
    cv_params = {
        # 'n_estimators': range(925, 975, 50),
        # 'max_depth': range(4, 9, 1),
        # 'min_child_weight': range(4, 9, 1),
        # 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
        # 'colsample_bytree': [0.6, 0.7, 0.8],
        # 'gamma': [0.2, 0.1],
        # 'reg_alpha': [2, 3, 4, 5, 6],
        # 'reg_lambda': [2, 3, 4, 5, 6, 7],
        # 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1]
    }
    model = xgb.XGBClassifier(
        learning_rate=0.001,
        n_estimators=925,
        max_depth=10,
        min_child_weight=3,
        seed=0,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=3,
        reg_lambda=1,
        metrics='auc',
        objective='binary:logistic')

    optimized_GBM = GridSearchCV(
        estimator=model,
        param_grid=cv_params,
        scoring=score,
        cv=5,
        verbose=1,
        n_jobs=4)

    optimized_GBM.fit(X_train, Y_train)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


def model_process(X_train, y_train, X_test):
    model = xgb.XGBClassifier(
        learning_rate=0.001,
        n_estimators=925,
        max_depth=10,
        min_child_weight=3,
        seed=0,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=3,
        reg_lambda=1,
        metrics='auc',
        objective='binary:logistic')
    model.fit(X_train, y_train)
    # result = model.predict(X_test)
    result = model.predict_proba(X_test)[:, 1]
    # print(result)
    submit = pd.read_csv('data/sample_submit.csv')
    submit['Evaluation'] = result
    submit.to_csv('submit_pro.csv', index=None)


if __name__ == "__main__":
    print("****************** start **********************")
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    X_train = train_data.drop(['CaseId', 'Evaluation'], axis=1)
    Y_train = train_data['Evaluation']
    X_test = test_data.drop(['CaseId'], axis=1)
    # xgb_cv(X_train, Y_train)
    model_process(X_train, Y_train, X_test)
