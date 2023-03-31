from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

from utils.DataSplitter import DataSplitter
from utils.MultiCategoryDataLoader import MultiCategoryDataLoader


joinedData = MultiCategoryDataLoader().get_data()
print('----------------------------------------------------------------------------')
for key in joinedData:
    print(key)
    X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data(joinedData=joinedData[key])
    clf = GaussianNB()
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    print('----------------------------------------------------------------------------')


