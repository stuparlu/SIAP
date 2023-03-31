from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from utils.DataSplitter import DataSplitter
from utils.MultiCategoryDataLoader import MultiCategoryDataLoader


joinedData = MultiCategoryDataLoader().get_data()
print('----------------------------------------------------------------------------')
for key in joinedData:
    print(key)
    X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data(joinedData=joinedData[key])

    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    Y_test = encoder.transform(Y_test)

    clf = xgb.XGBClassifier()
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)
    Y_pred = encoder.inverse_transform(Y_pred)
    Y_test = encoder.inverse_transform(Y_test)
    print(classification_report(Y_test, Y_pred, zero_division=1))

    xgb.plot_importance(clf, max_num_features=20)
    plt.show()
    print('----------------------------------------------------------------------------')


