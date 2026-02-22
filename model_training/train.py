import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import preprocess
from evaluate import evaluate


def main():
    # Load dataset
    df = pd.read_csv("train.csv")

    # Apply preprocessing
    X, y = preprocess(df)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize model
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    evaluate(model, X_test, y_test)

    # Save model to backend folder
    joblib.dump(model, "../backend/models/model.pkl")

    print("Model trained and saved successfully!")


if __name__ == "__main__":
    main()
