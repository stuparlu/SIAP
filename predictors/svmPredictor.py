from sklearn import svm
from sklearn.metrics import classification_report
from utils.DataSplitter import DataSplitter

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

clf = svm.SVC(kernel='linear', cache_size=7000)
clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)
print(classification_report(Y_test, y_pred))