import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from utils.dataWrangling.DataSplitter import DataSplitter
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.transform(Y_test)

clf = xgb.XGBClassifier()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
Y_pred = encoder.inverse_transform(Y_pred)
Y_test = encoder.inverse_transform(Y_test)

# Print the classification report
print(classification_report(Y_test, Y_pred, zero_division=1))

xgb.plot_importance(clf, max_num_features=20)
plt.show()
