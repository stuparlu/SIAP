import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from utils.dataWrangling.DataSplitter import DataSplitter
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.transform(Y_test)

clf = xgb.XGBClassifier()
# params= {'colsample_bytree': 0.9, 'gamma': 0.1, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.8}
# clf = xgb.XGBClassifier(**params)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
Y_pred = encoder.inverse_transform(Y_pred)
Y_test = encoder.inverse_transform(Y_test)

# Print the classification report
print(classification_report(Y_test, Y_pred, zero_division=1))

# xgb.plot_importance(clf, max_num_features=20)
# plt.show()

# param_grid = {
#     'max_depth': [3, 4, 5],
#     'learning_rate': [0.1, 0.01, 0.001],
#     'n_estimators': [50, 100, 200],
#     'gamma': [0, 0.1, 0.2],
#     'subsample': [0.8, 0.9],
#     'colsample_bytree': [0.8, 0.9]
# }
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
# grid_search.fit(X_train, Y_train)
#
# # Print the best hyperparameters
# print("Best hyperparameters:", grid_search.best_params_)