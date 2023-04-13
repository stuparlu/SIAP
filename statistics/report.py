from utils.dataWrangling.DataLoader import DataLoader

mainData = DataLoader().get_data()
# mainData.drop(mainData[mainData['state'] == 'successful'].index, inplace=True)
# mainData.drop(mainData[mainData['state'] == 'failed'].index, inplace=True)

print(mainData.keys())


categories = list(set(mainData['main_category'].values))

print(f"Project categories: {categories}")
print(50*'-')
print(f"Project count by category:")
for key in categories:
    count = mainData['main_category'].value_counts()[key]
    print(f"{key}: {count}")
print(50*'-')

# hasHeaderVideo = list(set(mainData['hasHeaderVideo'].values))
#
# print(f"Project Has header Video: {hasHeaderVideo}")
# print(50*'-')
# print(f"Project count by hasHeaderVideo:")
# for key in hasHeaderVideo:
#     count = mainData['hasHeaderVideo'].value_counts()[key]
#     print(f"{key}: {count}")
# print(50*'-')


states = list(set(mainData['state'].values))
print(f"Project states: {states}")
print(50*'-')
print(f"Project count by state:")
for key in states:
    count = mainData['state'].value_counts()[key]
    print(f"{key}: {count}")
print(50*'-')

numerical_values = ['goal', 'videoLength', 'descriptionMediaNumber', 'textLength', 'textReadingEase', 'imageTextRatio', 'projectDuration']
for value in numerical_values:
    print(f"Stats for {value}:")
    print(50*'-')
    print(f"Mean value: {mainData[value].mean()}")
    print(f"Median value: {mainData[value].median()}")
    print(f"Standard deviation value: {mainData[value].std()}")
    print(f"Minimum value: {mainData[value].min()}")
    print(f"Maximum value: {mainData[value].max()}")
    print(f"Variance value: {mainData[value].var()}")
    # print(f"Mode value: {mainData[value].mode()}")
    print(50*'-')


print(f"Text reading ease lower than threshold count: {(mainData['textReadingEase'] < 25).sum()}")
print(f"Text length lower than threshold project count: {(mainData['textLength'] < 300).sum()}")
print(f"Project duration lower than threshold project count: {(mainData['projectDuration'] < 5).sum()}")
print(f"Projects with goal lower than desired: {(mainData['goal'] < 100).sum()}")
count = len(mainData[(mainData['hasHeaderVideo'] == True) & (mainData['videoLength'] == 0)])
print(f"Projects with invalid header video: {count}")
