# Split the data
from sklearn.model_selection import train_test_split
from utils.DataLoader import DataLoader

class DataSplitter(object):
    def __init__(self):
        pass
    def get_splitted_data(self):
        joinedData = DataLoader().get_data()
        X = joinedData.drop(columns=['state'])
        y = joinedData['state']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.1, random_state=0)
        X_train = X_train.dropna()
        Y_train = Y_train[X_train.index.isin(X_train.index)]
        X_test = X_test.dropna()
        Y_test = Y_test[X_test.index.isin(X_test.index)]
        return X_train, X_test, Y_train, Y_test