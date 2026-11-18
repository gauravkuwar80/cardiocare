import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from components.data_preprocessing import build_preprocessor


datasets = {
    "Kaggle": "data/processed/heart_cleaned.csv",
    "Cleveland": "data/processed/heart_cleveland_cleaned.csv"
}

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

all_results = []

for dataset_name, path in datasets.items():

    print(f"\n{'='*50}")
    print("DATASET:", dataset_name)
    print("="*50)

    df = pd.read_csv(path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    for model_name, classifier in models.items():

        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-Score": f1_score(y_test, predictions, zero_division=0)
        }

        all_results.append(results)

        print(
            f"{model_name:22} | "
            f"Accuracy: {results['Accuracy']*100:.2f}% | "
            f"F1: {results['F1-Score']*100:.2f}%"
        )

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    "data/processed/dataset_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
print(results_df.to_string(index=False))