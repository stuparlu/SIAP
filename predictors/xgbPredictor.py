import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from utils.dataWrangling.DataSplitter import DataSplitter
import matplotlib.pyplot as plt

# param_grid = {
#     # 'max_depth': [None, 4],
#     'learning_rate': [None, 0.01],
#     'n_estimators': [50, 100, 200, 300],
#     'gamma': [None, 0.1],
#     'subsample': [None, 0.85, 0.9],
#     'colsample_bytree': [None, 0.8, 0.9, 0.95]
# }

param_grid = {
    'objective': ['binary:logistic',  'multi:softmax', 'reg:squarederror'],
    'use_label_encoder': [False],
    'base_score': [0.5, 0.75, 1.0],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'callbacks': [None],
    # 'colsample_bylevel': [0.5, 0.75, 1.0],
    # 'colsample_bynode': [0.5, 0.75, 1.0],
    # 'colsample_bytree': [0.5, 0.75, 1.0],
    'early_stopping_rounds': [None, 5, 10, 20],
    'enable_categorical': [False, True],
    # 'eval_metric': ['error', 'logloss', 'auc'],
    # 'feature_types': [None, 'auto', 'float'],
    # 'gamma': [0.0, 0.1, 0.5, 1],
    # 'gpu_id': [None, 0, 1],
    # 'grow_policy': ['depthwise', 'lossguide'],
    # 'importance_type': ['gain', 'cover', 'weight', 'total_gain', 'total_cover'],
    # 'interaction_constraints': [None, '[[1, 2], [3, 4]]', '[[0, 1, 2], [2, 3]]'],
    # 'learning_rate': [0.01, 0.1, 0.5],
    # 'max_bin': [256, 512, 1024],
    # 'max_cat_threshold': [16, 32, 64, 128],
    # 'max_cat_to_onehot': [4, 8, 16],
    # 'max_delta_step': [0, 1, 2],
    # 'max_depth': [3, 5, 7],
    # 'max_leaves': [None, 32, 64],
    # 'min_child_weight': [1, 3, 5, 10],
    # 'missing': [None, 0, -999],
    # 'monotone_constraints': [None],
    # 'n_estimators': [50, 100, 200],
    # 'n_jobs': [-1],
    # 'num_parallel_tree': [1, 2, 4],
    # 'predictor': [None, 'gpu_predictor', 'cpu_predictor'],
    # 'random_state': [42],
    # 'reg_alpha': [0, 0.1, 1],
    # 'reg_lambda': [0, 0.1, 1],
    'sampling_method': [None, 'uniform', 'gradient_based', 'hybrid'],
    # 'scale_pos_weight': [1, 2, 4, 10],
    'subsample': [0.5, 0.75, 1.0],
    'tree_method': ['auto', 'exact', 'approx', 'hist'],
    'validate_parameters': [ False, True],
    'verbosity': [0, 1, 2]
}

X_train, X_test, X_val, Y_val, Y_train, Y_test = DataSplitter().get_splitted_data()

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_val = encoder.transform(Y_val)
Y_test = encoder.transform(Y_test)

clf = xgb.XGBClassifier()
clf.fit(X_train, Y_train)
print(clf.get_params())

tunedClf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
tunedClf.fit(X_val, Y_val)
print(tunedClf.get_params())

print("Best hyperparameters:", tunedClf.best_params_)

Y_pred = tunedClf.predict(X_test)
Y_pred = encoder.inverse_transform(Y_pred)
Y_test = encoder.inverse_transform(Y_test)

# Print the classification report
print(classification_report(Y_test, Y_pred, zero_division=1))

xgb.plot_importance(clf, max_num_features=20)
plt.show()
