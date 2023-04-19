from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from utils.dataWrangling.DataSplitter import DataSplitter

# param_grid = {
#     'C': [1, 2, 5, 10, 15],
#     'kernel': ['rbf', 'poly', 'sigmoid'],
#     'gamma': ['scale', 'auto']
# }

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3],
    'gamma': ['scale', 'auto'],
    'shrinking': [True, False],
    'probability': [True, False],
    'tol': [1e-3, 1e-4],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000, 5000],
    'verbose': [0, 1, 2]
}

X_train, X_test, X_val, Y_val, Y_train, Y_test = DataSplitter().get_splitted_data()
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
X_val = preprocessing.scale(X_val)

clf = svm.SVC()
clf.fit(X_train, Y_train)
print(clf.get_params())


tunedClf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
tunedClf.fit(X_train, Y_train)
print(tunedClf.get_params())
print("Best hyperparameters:", tunedClf.best_params_)


Y_pred = tunedClf.predict(X_test)
print(classification_report(Y_test, Y_pred))
