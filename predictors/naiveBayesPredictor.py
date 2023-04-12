from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from utils.dataWrangling.DataSplitter import DataSplitter

X_train, X_test, X_val, Y_val, Y_train, Y_test = DataSplitter().get_splitted_data()

clf = GaussianNB()

# Perform a grid search to find the best hyperparameters
param_grid = {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]}
tunedClf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
tunedClf.fit(X_val, Y_val)
print("Best hyperparameters:", tunedClf.best_params_)

Y_pred = tunedClf.predict(X_test)
print(classification_report(Y_test, Y_pred))