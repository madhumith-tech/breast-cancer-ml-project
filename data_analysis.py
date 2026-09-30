import pandas as pd

data = pd.read_csv("data.csv")

print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())
print(data["diagnosis"].value_counts())