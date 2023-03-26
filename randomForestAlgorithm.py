from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
import textstat

def transform_description(description):
    return description.replace("\n", "").replace("\r", "")

def calculate_ease(description):
    return textstat.flesch_reading_ease(description);

def calculate_ratio(imageCount, textLength):
    if(imageCount == 0):
        return 0
    return textLength / imageCount

# Load the data
mainData = pd.read_csv('refinedDataSet.csv', index_col=False)

joinedData = mainData
joinedData.drop(joinedData[joinedData['state'] == 'canceled'].index, inplace=True)
joinedData.drop(joinedData[joinedData['state'] == 'live'].index, inplace=True)
joinedData.drop(joinedData[joinedData['state'] == 'undefined'].index, inplace=True)

# apply the function to all values in the 'City' column
joinedData['textDescription'] = joinedData['textDescription'].apply(transform_description)
joinedData["textReadingEase"] = joinedData['textDescription'].apply(calculate_ease)
joinedData['imageTextRatio'] = joinedData.apply(lambda row: calculate_ratio(row['descriptionMediaNumber'], row['textLength']), axis=1)
#print(joinedData['imageTextRatio'].values)
# print(joinedData["textReadingEase"].values)

# Drop text columns
joinedData = joinedData.drop(columns=['name', 'textDescription'])

# Encode categorical variables
categorical_cols = ['category', 'main_category', 'currency', 'country']
for col in categorical_cols:
    encoder = LabelEncoder()
    joinedData[col] = encoder.fit_transform(joinedData[col])


joinedData['launched'] = joinedData['launched'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))))
joinedData['deadline'] = joinedData['deadline'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d'))))
joinedData = joinedData.drop('ID', axis=1)
joinedData = joinedData.drop('Unnamed: 0', axis=1)

joinedData = joinedData.drop('backers', axis=1)
joinedData = joinedData.drop('goal', axis=1)
joinedData = joinedData.drop('pledged', axis=1)
joinedData = joinedData.drop('usd pledged', axis=1)
joinedData = joinedData.drop('usd_pledged_real', axis=1)

# Split the data
X = joinedData.drop(columns=['state'])
y = joinedData['state']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=0)

X_train_dropped = X_train.dropna()
y_train_dropped = y_train[X_train.index.isin(X_train_dropped.index)]

print(X_train_dropped.keys())

X_test_dropped = X_test.dropna()
y_test_dropped = y_test[X_test.index.isin(X_test_dropped.index)]

clf = RandomForestClassifier()

clf.fit(X_train_dropped, y_train_dropped)

# Make predictions
y_pred = clf.predict(X_test_dropped)

print(classification_report(y_test_dropped, y_pred))