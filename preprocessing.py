import pandas as pd

data = pd.read_csv("data.csv")

# Remove unnecessary columns
data = data.drop(["id", "Unnamed: 32"], axis=1)

# Convert diagnosis into numbers
data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

print(data.head())
print(data.shape)
print(data.isnull().sum())