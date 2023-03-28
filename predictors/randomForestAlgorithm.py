from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from utils.DataSplitter import DataSplitter

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

clf = RandomForestClassifier()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_pred))