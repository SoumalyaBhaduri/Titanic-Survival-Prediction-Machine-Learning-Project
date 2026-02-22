import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)


def evaluate(model, X_test, y_test):
    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    print("\nModel Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save metrics to JSON
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("artifacts/roc_curve.png")
    plt.close()

    print("\nEvaluation artifacts saved in 'artifacts/' folder.")
