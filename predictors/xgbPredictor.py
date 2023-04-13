import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from utils.dataWrangling.DataSplitter import DataSplitter
import matplotlib.pyplot as plt

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.5],
    'subsample': [0.5, 0.75, 1.0],
    'colsample_bytree': [0.5, 0.75, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1],
    'gamma': [0.0, 0.1, 0.5, 1],
}

X_train, X_test, X_val, Y_val, Y_train, Y_test = DataSplitter().get_splitted_data()

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_val = encoder.transform(Y_val)
Y_test = encoder.transform(Y_test)

clf = xgb.XGBClassifier()
clf.fit(X_train, Y_train)
print(clf.get_params())

tunedClf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
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
