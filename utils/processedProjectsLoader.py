import json
import pandas as pd

processedData = []

with open("../files/joinedTitles.json", "r", encoding='utf-8') as processedTitles:
    data = json.load(processedTitles)
    for project in data:
        if project.get("processed") is True:
            processedData.append(project)

# Load the data
mainData = pd.read_csv('../files/dataset.csv', index_col=False)



syntheticFrame = pd.DataFrame(processedData)
syntheticFrame.rename(columns={'title': 'name'}, inplace=True)
syntheticFrame = syntheticFrame.drop('processed', axis=1)
print(syntheticFrame.keys())


merged = pd.merge(mainData, syntheticFrame, on="name")
merged.to_csv("refinedDataSet.csv")




















# for project in processedData:



    # newFrame = pd.merge(row, syntheticRow, on="name")
    # print(newFrame)
    # row["hasHeaderVideo"] = project.get("hasHeaderVideo")

    # print(row.values)
    # refinedData = refinedData.append(row, ignore_index=True)

    # "hasHeaderVideo": false,
    # "videoLength": 0,
    # "descriptionMediaNumber": 0,
    # "textLength": 261,
    # "textDescription":
