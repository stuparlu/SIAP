import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



from utils.DataLoader import DataLoader

joinedData = DataLoader().get_data()

X = joinedData.drop(columns=['state'])
y = joinedData['state']
encoder = LabelEncoder()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=0)
X_train_dropped = X_train.dropna()
y_train_dropped = y_train[X_train.index.isin(X_train_dropped.index)]
X_test_dropped = X_test.dropna()
y_test_dropped = y_test[X_test.index.isin(X_test_dropped.index)]
y_train_encoded = encoder.fit_transform(y_train_dropped)
y_test_encoded = encoder.transform(y_test_dropped)

clf = xgb.XGBClassifier()
clf.fit(X_train_dropped, y_train_encoded)

# Make predictions
y_pred_encoded = clf.predict(X_test_dropped)
y_pred = encoder.inverse_transform(y_pred_encoded)

# Print the classification report
print(classification_report(y_test_dropped, y_pred, zero_division=1))