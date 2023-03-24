import json

with open("titles.json", "r") as file1:
    data = json.load(file1)
    sorted = []
    for title in data['data']:
        dict = {'title': title, 'processed': False, 'hasHeaderVideo': False, 'videoLength': 0, 'descriptionMediaNumber': 0, 'textLength': 0, 'textDescription': ''}
        sorted.append(dict)
    # dict = {"data": sorted.tolist()}

    with open("titles2.json", "w") as file2:
        file2.write(json.dumps(sorted))
