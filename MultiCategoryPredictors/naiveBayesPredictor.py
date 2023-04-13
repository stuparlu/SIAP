from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

from utils.dataWrangling.DataSplitter import DataSplitter
from utils.dataWrangling.MultiCategoryDataLoader import MultiCategoryDataLoader


joinedData = MultiCategoryDataLoader().get_data()
print('----------------------------------------------------------------------------')
for key in joinedData:
    print(key)
    X_train, X_test, X_val, Y_val, Y_train, Y_test = DataSplitter().get_splitted_data(joinedData=joinedData[key])
    clf = GaussianNB()
    clf.fit(X_train, Y_train)

    param_grid = {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]}
    tunedClf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
    tunedClf.fit(X_val, Y_val)
    print(f"Best hyperparameters for category: {key} --> {tunedClf.best_params_}")

    Y_pred = tunedClf.predict(X_test)
    print(classification_report(Y_test, Y_pred, zero_division=1))
    print('----------------------------------------------------------------------------')


