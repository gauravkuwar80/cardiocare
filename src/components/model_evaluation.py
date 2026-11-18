
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

from src.components.data_preprocessing import load_data


DATA_PATH = "data/processed/heart_cleaned.csv"
MODEL_PATH = "models/logistic_regression_optimized.pkl"
OUTPUT_DIR = "reports/model_evaluation"


def evaluate_model():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading dataset and trained model...")

    X, y = load_data(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # Reproduce the same split used during optimization
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Generating predictions...")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "F1-Score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "ROC-AUC": float(
            roc_auc_score(y_test, probabilities)
        ),
        "Average Precision": float(
            average_precision_score(y_test, probabilities)
        )
    }

    # Save metrics
    with open(
        os.path.join(OUTPUT_DIR, "evaluation_metrics.json"),
        "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    # Save classification report
    report = classification_report(
        y_test,
        predictions,
        target_names=["No Heart Disease", "Heart Disease"],
        zero_division=0
    )

    with open(
        os.path.join(OUTPUT_DIR, "classification_report.txt"),
        "w"
    ) as file:
        file.write(report)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    plt.figure(figsize=(7, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks([0, 1], ["Predicted Negative", "Predicted Positive"])
    plt.yticks([0, 1], ["Actual Negative", "Actual Positive"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        dpi=300
    )
    plt.close()

    # 2. ROC-AUC Curve
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "roc_curve.png"),
        dpi=300
    )
    plt.close()

    # 3. Precision-Recall Curve
    precision_values, recall_values, thresholds = (
        precision_recall_curve(y_test, probabilities)
    )

    avg_precision = average_precision_score(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(
        recall_values,
        precision_values,
        label=f"Average Precision = {avg_precision:.3f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "precision_recall_curve.png"),
        dpi=300
    )
    plt.close()

    # 4. Logistic Regression Coefficient Analysis
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute Coefficient": np.abs(coefficients)
    })

    feature_df = feature_df.sort_values(
        by="Absolute Coefficient",
        ascending=False
    )

    feature_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_coefficients.csv"),
        index=False
    )

    top_features = feature_df.head(15).sort_values(
        by="Absolute Coefficient"
    )

    plt.figure(figsize=(9, 7))
    plt.barh(
        top_features["Feature"],
        top_features["Coefficient"]
    )

    plt.xlabel("Logistic Regression Coefficient")
    plt.title("Top 15 Features by Absolute Coefficient")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_coefficients.png"),
        dpi=300
    )
    plt.close()

    # Print results
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nEvaluation files saved in:", OUTPUT_DIR)
    print("\nMODEL EVALUATION COMPLETED!")


if __name__ == "__main__":
    evaluate_model()
