from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from utils.dataWrangling.DataSplitter import DataSplitter
from utils.dataWrangling.MultiCategoryDataLoader import MultiCategoryDataLoader

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.5],
    'subsample': [0.5, 0.75, 1.0],
    'colsample_bytree': [0.5, 0.75, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1],
    'gamma': [0.0, 0.1, 0.5, 1],
}

joinedData = MultiCategoryDataLoader().get_data()
print('----------------------------------------------------------------------------')
for key in joinedData:
    print(key)
    X_train, X_test, X_val, Y_val, Y_train, Y_test = DataSplitter().get_splitted_data(joinedData=joinedData[key])

    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    Y_val = encoder.transform(Y_val)
    Y_test = encoder.transform(Y_test)

    clf = xgb.XGBClassifier()
    clf.fit(X_train, Y_train)

    tunedClf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
    tunedClf.fit(X_val, Y_val)
    print(f"Best hyperparameters for category: {key} --> {tunedClf.best_params_}")

    Y_pred = tunedClf.predict(X_test)
    Y_pred = encoder.inverse_transform(Y_pred)
    Y_test = encoder.inverse_transform(Y_test)
    print(classification_report(Y_test, Y_pred, zero_division=1))

    # xgb.plot_importance(tunedClf, max_num_features=20)
    plt.show()
    print('----------------------------------------------------------------------------')


