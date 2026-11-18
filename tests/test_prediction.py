
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score


MODEL_PATH = "models/logistic_regression_optimized.pkl"


def get_model():
    return joblib.load(MODEL_PATH)


def test_model_loads():
    model = get_model()
    assert model is not None


def test_model_prediction():
    model = get_model()

    sample = pd.DataFrame([{
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3
    }])

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1]


def test_model_accuracy():
    model = get_model()

    df = pd.read_csv("data/processed/heart_cleaned.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.70
