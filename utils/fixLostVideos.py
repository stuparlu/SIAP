import json

with open("../files/multiTitles/joinedTitles.json", "rb") as file1:
    file_contents = file1.read().decode('utf-8')
    allData = json.loads(file_contents)

with open("../files/multiTitles/titlesToAdd.json", "rb") as file1:
    file_contents = file1.read().decode('utf-8')
    dataToAdd = json.loads(file_contents)

for unfIndex, unfData in enumerate(allData):
    for addIndex, addData in enumerate(dataToAdd):
        if unfData.get('title') == addData.get('title'):
            allData[unfIndex] = addData
            print(addData)

with open("../files/multiTitles/fixed.json", "w") as file2:
    file2.write(json.dumps(allData, indent=4))