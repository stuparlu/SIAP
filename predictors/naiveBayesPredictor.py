from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from utils.dataWrangling.DataSplitter import DataSplitter

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}


clf = GaussianNB()

# Perform a grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

clf = GaussianNB(**grid_search.best_params_)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_pred))