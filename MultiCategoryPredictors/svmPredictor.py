from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from utils.dataWrangling.DataSplitter import DataSplitter
from utils.dataWrangling.MultiCategoryDataLoader import MultiCategoryDataLoader


joinedData = MultiCategoryDataLoader().get_data()
print('----------------------------------------------------------------------------')
for key in joinedData:
    print(key)
    X_train, X_test, X_val, Y_val, Y_train, Y_test = DataSplitter().get_splitted_data(joinedData=joinedData[key])
    clf = svm.SVC()
    clf.fit(X_train, Y_train)

    param_grid = {
        'C': [1, 2, 5, 10, 15],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
    tunedClf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
    tunedClf.fit(X_val, Y_val)
    print(f"Best hyperparameters for category: {key} --> {tunedClf.best_params_}")

    Y_pred = tunedClf.predict(X_test)
    print(classification_report(Y_test, Y_pred, zero_division=1))
    print('----------------------------------------------------------------------------')


