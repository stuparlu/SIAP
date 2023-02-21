import pandas as pd
df = pd.read_csv('dataset.csv', index_col=0, encoding='unknown-8bit')
df.hist(column='category')

