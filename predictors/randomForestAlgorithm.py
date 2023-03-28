from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.DataLoader import DataLoader

joinedData = DataLoader().get_data()

# Split the data
X = joinedData.drop(columns=['state'])
y = joinedData['state']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=0)
X_train_dropped = X_train.dropna()
y_train_dropped = y_train[X_train.index.isin(X_train_dropped.index)]
X_test_dropped = X_test.dropna()
y_test_dropped = y_test[X_test.index.isin(X_test_dropped.index)]

clf = RandomForestClassifier()
clf.fit(X_train_dropped, y_train_dropped)

# Make predictions
y_pred = clf.predict(X_test_dropped)

print(classification_report(y_test_dropped, y_pred))