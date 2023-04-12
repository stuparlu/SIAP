import time
from datetime import datetime

import textstat
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


        joinedData = mainData
        joinedData.drop(joinedData[joinedData['state'] == 'canceled'].index, inplace=True)
        joinedData.drop(joinedData[joinedData['state'] == 'live'].index, inplace=True)
        joinedData.drop(joinedData[joinedData['state'] == 'undefined'].index, inplace=True)

        joinedData['textDescription'] = joinedData['textDescription'].apply(transform_description)
        # print(joinedData['textLength'])
        joinedData['textLength'] = joinedData['textDescription'].apply(lambda x: len(x))
        # print(joinedData['textLength'])
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

        joinedData = joinedData.drop(joinedData[joinedData['goal'] < 100].index)
        joinedData = joinedData.drop(joinedData[joinedData['projectDuration'] < 5].index)
        joinedData = joinedData.drop(joinedData[joinedData['textLength'] < 300].index)
        joinedData = joinedData.drop(joinedData[joinedData['textReadingEase'] < 25].index)

        joinedData = joinedData.drop(columns=['ID','Unnamed: 0', 'backers', 'pledged', 'usd_goal_real', 'usd pledged',
                                              'usd_pledged_real', 'name', 'textDescription', 'launched'])
        return joinedData