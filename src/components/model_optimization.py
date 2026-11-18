
import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.components.data_preprocessing import (
    load_data,
    build_preprocessor
)


DATA_PATH = "data/processed/heart_cleaned.csv"
MODEL_DIR = "models"
RESULTS_DIR = "data/processed"
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "CardioCare_Model_Optimization"


def get_model_grids():

    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=42),
            {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__solver": ["liblinear", "lbfgs"]
            }
        ),

        "Naive Bayes": (
            GaussianNB(),
            {
                "classifier__var_smoothing": [
                    1e-11, 1e-10, 1e-9, 1e-8, 1e-7
                ]
            }
        ),

        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 5, 10, 15],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2]
            }
        ),

        "Gradient Boosting": (
    GradientBoostingClassifier(random_state=42),
    {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [2, 3],
        "classifier__subsample": [0.8, 1.0]
    }
),


"Extra Trees": (
    ExtraTreesClassifier(random_state=42),
    {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2]
    }
),

"AdaBoost": (
    AdaBoostClassifier(random_state=42),
    {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1, 1.0]
    }
),

        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=1
            ),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [2, 3, 4],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__subsample": [0.8, 1.0]
            }
        ),

        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2]
            }
        ),

        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "classifier__max_depth": [None, 3, 5, 7, 10],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__criterion": ["gini", "entropy"]
            }
        ),

        "Support Vector Machine": (
            SVC(random_state=42),
            {
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__kernel": ["linear", "rbf"],
                "classifier__gamma": ["scale", "auto", 0.01, 0.1, 1]
            }
        )
    }


def optimize_models():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y = load_data(DATA_PATH)

    # Keep test data separate from hyperparameter tuning
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

    model_grids = get_model_grids()

    results = []
    best_models = {}

    
    for model_name, (model, parameters) in model_grids.items():

        with mlflow.start_run(run_name=model_name):

            print("\n" + "=" * 55)
            print(f"Optimizing: {model_name}")
            print("=" * 55)

            pipeline = Pipeline(steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", model)
            ])

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=parameters,
                scoring="f1",
                cv=cv,
                n_jobs=1,
                verbose=1,
                refit=True
            )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            predictions = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(
                y_test, predictions, zero_division=0
            )
            recall = recall_score(
                y_test, predictions, zero_division=0
            )
            f1 = f1_score(
                y_test, predictions, zero_division=0
            )

            # Log model parameters
            mlflow.log_params(search.best_params_)
            mlflow.log_param("algorithm", model_name)
            mlflow.log_param("cv_strategy", "RepeatedStratifiedKFold")
            mlflow.log_param("cv_splits", 5)
            mlflow.log_param("cv_repeats", 3)

            # Log performance metrics
            mlflow.log_metric("cv_best_f1", search.best_score_)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1", f1)

            # Log trained model
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model"
            )

            results.append({
                "Model": model_name,
                "CV Best F1": round(search.best_score_, 4),
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1, 4),
                "Best Parameters": str(search.best_params_)
            })

            best_models[model_name] = best_model

            model_filename = (
                model_name.lower()
                .replace(" ", "_")
                .replace("-", "_")
                + "_optimized.pkl"
            )

            joblib.dump(
                best_model,
                os.path.join(MODEL_DIR, model_filename)
            )

            print("Best CV F1:", round(search.best_score_, 4))
            print("Test Accuracy:", round(accuracy, 4))
            print("Best Parameters:", search.best_params_)


    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="CV Best F1",
        ascending=False
    )

    results_path = os.path.join(
        RESULTS_DIR,
        "optimized_model_comparison.csv"
    )

    results_df.to_csv(results_path, index=False)

    # Select model using cross-validation F1-score
    best_model_name = results_df.iloc[0]["Model"]
    best_model = best_models[best_model_name]

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "optimized_best_model.pkl")
    )

    # Save SVM separately
    joblib.dump(
        best_models["Support Vector Machine"],
        os.path.join(MODEL_DIR, "optimized_svm.pkl")
    )

    summary = {
        "best_model": best_model_name,
        "selection_metric": "5-fold Cross-validation F1-score",
        "cross_validation": "5-fold StratifiedKFold",
        "test_size": 0.20,
        "test_random_state": 42
    }

    with open(
        os.path.join(RESULTS_DIR, "optimization_summary.json"),
        "w"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 55)
    print("OPTIMIZATION COMPLETED")
    print("=" * 55)

    print(results_df.to_string(index=False))
    print("\nBest Model:", best_model_name)


if __name__ == "__main__":
    optimize_models()
