# Load the data
import json

import pandas as pd

mainData = pd.read_csv('../../files/refinedDataSet.csv', index_col=False)

mainData = mainData[mainData['state'] == 'successful']
values = mainData['name'].values

# print(dumps)
valuesList = values.tolist()

sorted = []
for name in values:
    dict = {'title': name, 'processed': False, 'hasHeaderVideo': False, 'videoLength': 0, 'descriptionMediaNumber': 0, 'textLength': 0, 'textDescription': ''}
    sorted.append(dict)

with open("../titles2.json", "w") as file2:
    file2.write(json.dumps(sorted))


