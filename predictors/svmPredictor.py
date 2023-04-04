from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from utils.dataWrangling.DataSplitter import DataSplitter

param_grid = {
    'C': [5, 10, 15],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

clf = svm.SVC()

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

clf = svm.SVC(**grid_search.best_params_)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_pred))
