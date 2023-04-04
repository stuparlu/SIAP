from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import preprocessing
from utils.dataWrangling.DataSplitter import DataSplitter

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_pred))
