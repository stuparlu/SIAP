from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 20, 30],
    'max_samples': [0.5, 1.0]
}

from utils.dataWrangling.DataSplitter import DataSplitter

X_train, X_test, X_val, Y_val, Y_train, Y_test = DataSplitter().get_splitted_data()

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
X_val = preprocessing.scale(X_val)

base_estimator = svm.SVC(kernel='rbf')
clf = BaggingClassifier(estimator=base_estimator)
clf.fit(X_train, Y_train)
print(clf.get_params())

tunedClf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
tunedClf.fit(X_val, Y_val)
print(tunedClf.get_params())
print("Best hyperparameters:", tunedClf.best_params_)


Y_pred = tunedClf.predict(X_test)
print(classification_report(Y_test, Y_pred))
