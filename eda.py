import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data.csv")

data = data.drop(["id", "Unnamed: 32"], axis=1)

data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

# 1. Diagnosis count
sns.countplot(x="diagnosis", data=data)
plt.title("Benign vs Malignant")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# 3. Radius comparison
sns.boxplot(x="diagnosis", y="radius_mean", data=data)
plt.title("Radius Mean vs Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Radius Mean")
plt.show()