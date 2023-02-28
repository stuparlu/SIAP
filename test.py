import pandas as pd
mainData = pd.read_csv('dataset.csv', index_col=False)
firstScrapedData = pd.read_csv('DataLuka.csv', index_col=False)
secondScrapedData = pd.read_csv('DataStefan.csv', index_col=False)

scrapedData = [firstScrapedData, secondScrapedData]
scrapedData = pd.concat(scrapedData)
print()

mainDataSubset = mainData[mainData['ID'].isin(scrapedData["ID"].tolist())]

joinedData = pd.merge(mainDataSubset, scrapedData, how='inner', on='ID')
print(joinedData.columns)
print(joinedData.values)