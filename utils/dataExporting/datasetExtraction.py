import json

import pandas as pd


# Load the data
mainData = pd.read_csv('../../files/dataset.csv', index_col=False)
values = mainData['name'].values

# print(dumps)
valuesList = values.tolist()
print(len(valuesList))
valuesList = [item for item in valuesList if not(pd.isnull(item)) == True]
print(len(valuesList))
valuesList = [item for item in valuesList if item != "NaN"]
print(len(valuesList))

#
# dict = {"data": values.tolist()}
#
# with open("titles.json", "w") as file1:
#     file1.write(json.dumps(dict))