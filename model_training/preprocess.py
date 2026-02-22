# preprocess.py
import pandas as pd


def preprocess(df: pd.DataFrame):
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]
    df = df.dropna()

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    return X, y
