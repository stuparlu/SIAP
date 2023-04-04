import time
from datetime import datetime

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

def calculate_project_duration(deadline, launched):
    # print(deadline," +++++ ", launched)
    deadline_date = datetime.strptime(deadline, '%Y-%m-%d')
    launched_date = datetime.strptime(launched, '%Y-%m-%d %H:%M:%S')
    delta = deadline_date - launched_date
    return delta.days

class DataLoader(object):
    def __init__(self):
        pass
    def get_data(self):
        # Load the data
        mainData = pd.read_csv('../files/refinedDataSet.csv', index_col=False)
        # print(mainData.count())

        joinedData = mainData
        joinedData.drop(joinedData[joinedData['state'] == 'canceled'].index, inplace=True)
        joinedData.drop(joinedData[joinedData['state'] == 'live'].index, inplace=True)
        joinedData.drop(joinedData[joinedData['state'] == 'undefined'].index, inplace=True)

        # apply the function to all values in the 'City' column
        joinedData['textDescription'] = joinedData['textDescription'].apply(transform_description)
        joinedData["textReadingEase"] = joinedData['textDescription'].apply(calculate_ease)
        joinedData['imageTextRatio'] = joinedData.apply(
            lambda row: calculate_ratio(row['descriptionMediaNumber'], row['textLength']), axis=1)
        joinedData['projectDuration'] = joinedData.apply(
            lambda row: calculate_project_duration(row['deadline'], row['launched']), axis=1)
        joinedData['launched'] = joinedData['launched'].apply(
            lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))))
        joinedData['deadline'] = joinedData['deadline'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d'))))
        joinedData = joinedData.drop_duplicates(subset="name")
        joinedData = joinedData.drop_duplicates(subset="ID")

        count_success = joinedData[joinedData['state'] == 'successful'].shape[0]
        count_failed = joinedData[joinedData['state'] == 'failed'].shape[0]
        min_count = min(count_success, count_failed)
        # Get a random sample of the data for each state, with size equal to the minimum count
        df_success = joinedData[joinedData['state'] == 'successful'].sample(n=min_count)
        df_failed = joinedData[joinedData['state'] == 'failed'].sample(n=min_count)
        joinedData = pd.concat([df_success, df_failed])
        joinedData = joinedData.sample(frac=1).reset_index(drop=True)

        joinedData = joinedData.drop(columns=['ID','Unnamed: 0', 'backers', 'pledged', 'usd_goal_real', 'usd pledged',
                                              'usd_pledged_real', 'name', 'textDescription', 'launched', 'hasHeaderVideo'])
        return joinedData