import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Load data
data = pd.read_csv("data.csv")

# Remove unnecessary columns
data = data.drop(["id", "Unnamed: 32"], axis=1)

# Convert diagnosis into numbers
data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

# Separate input and output
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]


# Create models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}


# Cross-validation
for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    print("\n", name)
    print("Scores:", scores)
    print("Average:", round(scores.mean(), 4))