from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from utils.dataWrangling.DataSplitter import DataSplitter

param_grid = {
    'n_estimators': [100, 50, 25],
    'max_depth': [None, 1],
    'min_samples_split': [7, 8, 6],
    'min_samples_leaf': [2, 3],
    'random_state': [42]
}

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# for i in range(len(grid_search.cv_results_['params'])):
#     print("Parameters:", grid_search.cv_results_['params'][i])
#     print("Mean test score:", grid_search.cv_results_['mean_test_score'][i])
#     print("Standard deviation test score:", grid_search.cv_results_['std_test_score'][i])
#     print()

# clf = RandomForestClassifier()
# clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
clf = RandomForestClassifier(**grid_search.best_params_)

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
