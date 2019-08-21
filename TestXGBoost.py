import self_def_functions.data_analytic as da
import joblib
import pickle
from collections import OrderedDict
from scipy.stats import skew
from xgboost import plot_importance
import xgboost as xgb
import csv
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

import os
os.environ["PATH"] += os.pathsep + 'C:\Graphviz2.38\\release\\bin'


# # [0] load data
# train_path = "dt0717-2-pk_by_city_pkw\\train_oneHot.csv"
# test_path = "dt0717-2-pk_by_city_pkw\\test_oneHot.csv"

# train = pd.read_csv(train_path, encoding='utf8')
# test = pd.read_csv(test_path, encoding='utf8')
# # test['total_price'] = np.NaN

train.info(memory_usage='deep')
test.info(memory_usage='deep')


print('[1] transform int col...')
tStart = time.time()  # 計時開始
int_col_list = list(train.select_dtypes(include=['int']).columns.values)
train[int_col_list] = train[int_col_list].apply(
    pd.to_numeric, downcast='unsigned')
test[int_col_list] = test[int_col_list].apply(
    pd.to_numeric, downcast='unsigned')
tEnd = time.time()
print("It cost %f sec" % (tEnd - tStart))
# train[int_col_list].dtypes


print('[2] transform float col...')
tStart = time.time()  # 計時開始
flt_col_list = list(train.select_dtypes(include=['float']).columns.values)
train[flt_col_list] = train[flt_col_list].apply(
    pd.to_numeric, downcast='float')
test[flt_col_list] = test[flt_col_list].apply(pd.to_numeric, downcast='float')
tEnd = time.time()
print("It cost %f sec" % (tEnd - tStart))
# train[flt_col_list].dtypes

print('[3] transform category col...')
tStart = time.time()  # 計時開始
cat_col_list = list(train.select_dtypes(include=['category']).columns.values)
train[cat_col_list] = train[cat_col_list].astype('uint8')
test[cat_col_list] = test[cat_col_list].astype('uint8')
tEnd = time.time()
print("It cost %f sec" % (tEnd - tStart))


# [1] 分訓練測試集

# 去掉無用的欄位
tag_list = ['fill_village_income_median', 'fill_txn_floor',
            'fill_parking_price', 'fill_parking_area', 'err_city_town', 'err_town']
train.drop(tag_list, axis=1, inplace=True)
test.drop(tag_list, axis=1, inplace=True)

dismiss_list = ['building_id']
train.drop(dismiss_list, axis=1, inplace=True)
test.drop(dismiss_list, axis=1, inplace=True)

train.info(memory_usage='deep')
test.info(memory_usage='deep')

# 標記目標欄位
label_col = 'total_price'

train_x = train[list(train.columns[~train.columns.isin([label_col])].values)]
train_y = train[label_col]
test_x = test[list(test.columns[~test.columns.isin([label_col])].values)]
test_y = test[label_col]
print('Training X Shape:', train_x.shape)
print('Training Y Shape:', train_y.shape)
print('Testing X Shape:', test_x.shape)
print('Testing Y Shape:', test_y.shape)


dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)


print("Ready!!!!!")
importance_type = 'total_gain'

# [2] model train
# learning_rate每次學多少 n_estimators每次學多少次，先設小一點試
model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                         colsample_bynode=1, colsample_bytree=0.8, gamma=0,
                         importance_type=importance_type, learning_rate=0.1, max_delta_step=0,
                         max_depth=10, min_child_weight=1, missing=None, n_estimators=1000,
                         n_jobs=1, nthread=7, objective='reg:linear', random_state=0,
                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=123,
                         silent=0, subsample=0.9, verbosity=1)

print('[4] first model train to pick up features...')
tStart = time.time()  # 計時開始
model.fit(train_x, train_y, eval_metric=da.hitrate,
          eval_set=[(train_x, train_y), (test_x, test_y)])
tEnd = time.time()
print("It cost %f sec" % (tEnd - tStart))


ax = xgb.plot_importance(model, max_num_features=400,
                         height=1, importance_type=importance_type)  # 找到前400的total gain
fig = ax.figure
xgb.plot_importance(model, max_num_features=400, height=0.5, ax=ax)
plt.show()


# 把重要性排序出來
Feature_Dict = model.get_booster().get_score(importance_type=importance_type)
dt_feature = pd.DataFrame.from_dict(Feature_Dict, orient='index')
dt_feature = dt_feature.rename(columns={'index': 'features', 0: 'importance'})
dt_feature = dt_feature.sort_values(by='importance', ascending=False)
pick_features = dt_feature.ix[:400, :].reset_index()

# plt trees
# digraph = xgb.to_graphviz(model.get_booster(), num_trees=1)
# digraph.format = 'png'
# digraph.view('./hhh')

most_relevant_features = list(pick_features.ix[:, 0])

# 這段為第一次跑LR=0.1。保留做比較用

print('[5] first model train use most relevent features...')
tStart = time.time()  # 計時開始
# 用total_gain來跑400
xgb_select_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                    colsample_bynode=1, colsample_bytree=0.8, gamma=0,
                                    importance_type='total_gain', learning_rate=0.005, max_delta_step=0,
                                    max_depth=10, min_child_weight=1, missing=None, n_estimators=100000,
                                    n_jobs=1, nthread=7, objective='reg:linear', random_state=0,
                                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=123,
                                    silent=0, subsample=0.9, verbosity=1)
xgb_select_model.fit(train_x[most_relevant_features], train_y, eval_metric=da.hitrate, eval_set=[
                     (train_x[most_relevant_features], train_y), (test_x[most_relevant_features], test_y)])

# save model
# Both functions save_model and dump_model save the model, the difference is that in dump_model you can save feature name and save tree in text format.

# The model can be saved.
xgb_select_model.save_model('model\\0719-2_100000_0_005.model')
# # The model and its feature map can also be dumped to a text file.
# xgb_select_model.dump_model('model\\0719_50000_1_10_dump.raw.txt')  # dump model
# xgb_select_model.dump_model('model\\0719_50000_1_10_dump.raw.txt', 'model\\0719_50000_1_10_featmap.txt')

# A saved model can be loaded as follows:
# init model
# xgb_select_model = xgb.Booster().load_model("model\\0719_50000_1_10.model")
# pickle.dump(xgb_select_model, open(
#     "model\\test.dat", "wb"))
tEnd = time.time()
print("It cost %f sec" % (tEnd - tStart))
# before feature selection
# y_hats = model.predict(
#     test_x, validate_features=True)
# after feature selection
y_hats = xgb_select_model.predict(
    test_x[most_relevant_features], validate_features=True)

submit_path = "org\\submit_test.csv"
submit = pd.read_csv(submit_path, encoding='utf8')


df_answers = pd.DataFrame()
df_answers['building_id'] = submit['building_id']
df_answers['total_price'] = list(y_hats)
df_answers
# 至此是submit test 的新資料答案

df_answers.to_csv(
    'results\\0719-2_100000_0_005.csv', index=False)
