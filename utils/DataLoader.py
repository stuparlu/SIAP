import time
import textstat
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def transform_description(description):
    return description.replace("\n", "").replace("\r", "")
def calculate_ease(description):
    return textstat.flesch_reading_ease(description);
def calculate_ratio(imageCount, textLength):
    if (imageCount == 0):
        return 0
    return textLength / imageCount
class DataLoader(object):
    def __init__(self):
        pass
    def get_data(self):
        # Load the data
        mainData = pd.read_csv('../files/refinedDataSet.csv', index_col=False)

        joinedData = mainData
        joinedData.drop(joinedData[joinedData['state'] == 'canceled'].index, inplace=True)
        joinedData.drop(joinedData[joinedData['state'] == 'live'].index, inplace=True)
        joinedData.drop(joinedData[joinedData['state'] == 'undefined'].index, inplace=True)

        # apply the function to all values in the 'City' column
        joinedData['textDescription'] = joinedData['textDescription'].apply(transform_description)
        joinedData["textReadingEase"] = joinedData['textDescription'].apply(calculate_ease)
        joinedData['imageTextRatio'] = joinedData.apply(
            lambda row: calculate_ratio(row['descriptionMediaNumber'], row['textLength']), axis=1)
        # Drop text columns
        joinedData = joinedData.drop(columns=['name', 'textDescription'])

        # Encode categorical variables
        categorical_cols = ['category', 'main_category', 'currency', 'country']
        for col in categorical_cols:
            encoder = LabelEncoder()
            joinedData[col] = encoder.fit_transform(joinedData[col])

        joinedData['launched'] = joinedData['launched'].apply(
            lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))))
        joinedData['deadline'] = joinedData['deadline'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d'))))
        joinedData = joinedData.drop('ID', axis=1)
        joinedData = joinedData.drop('Unnamed: 0', axis=1)

        joinedData = joinedData.drop('backers', axis=1)
        joinedData = joinedData.drop('goal', axis=1)
        joinedData = joinedData.drop('pledged', axis=1)
        joinedData = joinedData.drop('usd pledged', axis=1)
        joinedData = joinedData.drop('usd_pledged_real', axis=1)
        return joinedData