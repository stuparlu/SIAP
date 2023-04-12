# Split the data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler


from utils.dataWrangling.DataLoader import DataLoader

class DataSplitter(object):
    def __init__(self):
        pass
    def get_splitted_data(self, joinedData = None):
        if joinedData is None:
            joinedData = DataLoader().get_data()

        # Encode categorical variables
        categorical_cols = ['category', 'main_category', 'currency', 'country']
        for col in categorical_cols:
            encoder = LabelEncoder()
            joinedData[col] = encoder.fit_transform(joinedData[col])

        X = joinedData.drop(columns=['state'])
        y = joinedData['state']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.3, random_state=0)
        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

        X_train = X_train.dropna()
        Y_train = Y_train[X_train.index.isin(X_train.index)]
        X_test = X_test.dropna()
        Y_test = Y_test[X_test.index.isin(X_test.index)]
        X_val = X_val.dropna()
        Y_val = Y_val[X_val.index.isin(X_val.index)]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

        rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_train, Y_train = rus.fit_resample(X_train, Y_train)

        return X_train, X_test, X_val, Y_val, Y_train, Y_test













        # count_success = joinedData[joinedData['state'] == 'successful'].shape[0]
        # count_failed = joinedData[joinedData['state'] == 'failed'].shape[0]
        # min_count = min(count_success, count_failed)
        # df_success = joinedData[joinedData['state'] == 'successful'].sample(n=min_count, random_state=42)
        # df_failed = joinedData[joinedData['state'] == 'failed'].sample(n=min_count, random_state=42)
        # joinedData = pd.concat([df_success, df_failed])
        # joinedData = joinedData.sample(frac=1, random_state=42).reset_index(drop=True)