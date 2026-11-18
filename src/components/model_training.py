
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from src.components.data_preprocessing import (
    load_data,
    build_preprocessor
)


# Project paths
DATA_PATH = "data/processed/heart_cleaned.csv"
MODEL_DIR = "models"
RESULTS_DIR = "data/processed"


def get_models():
    """Return all seven machine learning algorithms."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        "Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Support Vector Machine": SVC(
            probability=True,
            random_state=42
        )
    }

    return models


def train_and_evaluate():
    """Train and evaluate all seven machine learning models."""

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load dataset
    X, y = load_data(DATA_PATH)

    # Split data before preprocessing to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = get_models()

    results = []
    trained_models = {}

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        # Each model gets its own preprocessing pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model)
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        # Save evaluation results
        results.append({
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4)
        })

        trained_models[model_name] = pipeline

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

    # Create comparison table
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1-Score",
        ascending=False
    )

    results_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )

    results_df.to_csv(results_path, index=False)

    # Select best model using F1-score
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    # Save best model
    model_path = os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )

    joblib.dump(best_model, model_path)

    # Save confusion matrix and classification report
    best_predictions = best_model.predict(X_test)

    cm = confusion_matrix(y_test, best_predictions)

    with open(
        os.path.join(RESULTS_DIR, "best_model_report.json"),
        "w"
    ) as file:
        json.dump(
            {
                "best_model": best_model_name,
                "confusion_matrix": cm.tolist(),
                "classification_report": classification_report(
                    y_test,
                    best_predictions,
                    output_dict=True,
                    zero_division=0
                )
            },
            file,
            indent=4
        )

    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETED")
    print("=" * 50)

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    print(f"\nBest Model: {best_model_name}")
    print(f"Saved Model: {model_path}")
    print(f"Results Saved: {results_path}")

    return results_df


if __name__ == "__main__":
    train_and_evaluate()
