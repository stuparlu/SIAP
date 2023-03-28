from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import time
import numpy as np

# Load the data
mainData = pd.read_csv('../files/dataset.csv', index_col=False)
firstScrapedData = pd.read_csv('../files/DataLuka.csv', index_col=False)
secondScrapedData = pd.read_csv('../files/DataStefan.csv', index_col=False)

scrapedData = [firstScrapedData, secondScrapedData]
scrapedData = pd.concat(scrapedData)

# Merge the data
mainDataSubset = mainData[mainData['ID'].isin(scrapedData["ID"].tolist())]
joinedData = pd.merge(mainDataSubset, scrapedData, how='inner', on='ID')

joinedData.drop(joinedData[joinedData['state'] == 'canceled'].index, inplace=True)

# Drop text columns
joinedData = joinedData.drop(columns=['name', 'description'])

# Encode categorical variables
categorical_cols = ['category', 'main_category', 'currency', 'country']
for col in categorical_cols:
    encoder = LabelEncoder()
    joinedData[col] = encoder.fit_transform(joinedData[col])


joinedData['launched'] = joinedData['launched'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))))
joinedData['deadline'] = joinedData['deadline'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d'))))


# Split the data
X = joinedData.drop(columns=['state'])
y = joinedData['state']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# Train the model
clf = GaussianNB()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))