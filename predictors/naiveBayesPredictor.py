from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from utils.dataWrangling.DataSplitter import DataSplitter

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

clf = GaussianNB()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_pred))