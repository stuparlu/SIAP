from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from utils.dataWrangling.DataSplitter import DataSplitter

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

# clf = RandomForestClassifier()
# clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=10, min_samples_leaf=2, random_state=42)

clf.fit(X_train, Y_train)


Y_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_pred))

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# # Perform a grid search to find the best hyperparameters
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
# grid_search.fit(X_train, Y_train)
#
# # Print the best hyperparameters
# print("Best hyperparameters:", grid_search.best_params_)
