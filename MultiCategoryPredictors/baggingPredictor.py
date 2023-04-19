from sklearn.ensemble import BaggingClassifier
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
    base_estimator = svm.SVC(kernel='rbf')
    clf = BaggingClassifier(estimator=base_estimator)

    # param_grid = {
    #     'n_estimators': [10, 20, 30],
    #     'max_samples': [0.5, 1.0]
    # }
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 1.0],
        'max_features': [0.5, 1.0],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False],
        'n_jobs': [-1]
    }

    tunedClf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
    tunedClf.fit(X_val, Y_val)
    print(f"Best hyperparameters for category: {key} --> {tunedClf.best_params_}")

    Y_pred = tunedClf.predict(X_test)
    print(classification_report(Y_test, Y_pred, zero_division=1))
    print('----------------------------------------------------------------------------')


