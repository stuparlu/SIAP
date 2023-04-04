from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 20, 30],
    'max_samples': [0.5, 0.7, 1.0]
}

from utils.dataWrangling.DataSplitter import DataSplitter

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

base_estimator = svm.SVC(kernel='linear')
clf = BaggingClassifier(base_estimator=base_estimator)

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Create the bagging classifier with the best hyperparameters
clf = BaggingClassifier(base_estimator=base_estimator, **grid_search.best_params_)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_pred))
