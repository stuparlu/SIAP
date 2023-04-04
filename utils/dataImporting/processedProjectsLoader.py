import json
import pandas as pd



with open("../../files/multiTitles/titlesStefan.json", "rb") as processedTitles:
    file_contents = processedTitles.read().decode('utf-8')
    dataStefan = json.loads(file_contents)

with open("../../files/multiTitles/titlesLuka.json", "rb") as processedTitles:
    file_contents = processedTitles.read().decode('utf-8')
    dataLuka = json.loads(file_contents)

with open("../../files/multiTitles/titlesSuccess.json", "rb") as processedTitles:
    file_contents = processedTitles.read().decode('utf-8')
    successData = json.loads(file_contents)

processedData = []
allData = dataStefan + dataLuka
for project in allData:
    if project.get("processed") is True:
        processedData.append(project)

print(len(processedData))

for index, value in enumerate(processedData):
    for successIndex, successValue in enumerate(successData):
        if value.get('title') == successValue.get('title'):
            processedData[index] = successValue


mainData = pd.read_csv('../../files/dataset.csv', index_col=False)



syntheticFrame = pd.DataFrame(processedData)
syntheticFrame.rename(columns={'title': 'name'}, inplace=True)
syntheticFrame = syntheticFrame.drop('processed', axis=1)
print(syntheticFrame.keys())


merged = pd.merge(mainData, syntheticFrame, on="name")
merged.to_csv("refinedDataSet.csv")

#
#

















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
