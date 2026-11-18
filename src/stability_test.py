
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

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
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

all_results = []

for dataset_name, path in datasets.items():

    print("\n" + "=" * 55)
    print("DATASET:", dataset_name)
    print("=" * 55)

    df = pd.read_csv(path)

    X = df.drop(columns=["target"])
    y = df["target"]

    for model_name, classifier in models.items():

        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "f1"],
            n_jobs=1
        )

        accuracy_mean = scores["test_accuracy"].mean()
        accuracy_std = scores["test_accuracy"].std()

        f1_mean = scores["test_f1"].mean()
        f1_std = scores["test_f1"].std()

        all_results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Mean Accuracy": accuracy_mean,
            "Accuracy Std": accuracy_std,
            "Mean F1": f1_mean,
            "F1 Std": f1_std
        })

        print(
            f"{model_name:22} | "
            f"Accuracy: {accuracy_mean*100:.2f}% "
            f"(±{accuracy_std*100:.2f}) | "
            f"F1: {f1_mean*100:.2f}% "
            f"(±{f1_std*100:.2f})"
        )

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    "data/processed/stability_comparison.csv",
    index=False
)

print("\nStability results saved successfully!")
print("\n", results_df.to_string(index=False))
