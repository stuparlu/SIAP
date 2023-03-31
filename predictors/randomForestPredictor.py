import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils.DataSplitter import DataSplitter

X_train, X_test, Y_train, Y_test = DataSplitter().get_splitted_data()

clf = RandomForestClassifier()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_pred))

# importances = clf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
# feature_names = [f"feature {i}" for i in range(X_train.shape[1])]
#
# forest_importances = pd.Series(importances, index=feature_names)
#
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
#
# fig.show()
# fig.savefig('sampleFileName.png')
