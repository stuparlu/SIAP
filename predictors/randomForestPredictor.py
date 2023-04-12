from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from utils.dataWrangling.DataSplitter import DataSplitter

param_grid = {
    'n_estimators': [200, 150, 125, 100, 75, 50],
    'max_depth': [None, 1, 2],
    'min_samples_split': [2, 3, 6, 7],
    'min_samples_leaf': [1, 2, 3, 4],
    'random_state': [42]
}

X_train, X_test, X_val, Y_val, Y_train, Y_test = DataSplitter().get_splitted_data()

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, Y_train)
print(clf.get_params())

tunedClf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
tunedClf.fit(X_val, Y_val)
print("Best hyperparameters:", tunedClf.best_params_)

Y_pred = tunedClf.predict(X_test)
print(classification_report(Y_test, Y_pred))
