
import pandas as pd
from flask import Flask, render_template, request
from src.pipelines.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

# Load the trained prediction model
prediction_pipeline = PredictionPipeline()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():

    results = pd.read_csv(
        "data/processed/optimized_model_comparison.csv"
    )

    return render_template(
        "dashboard.html",
        models=results.to_dict(orient="records")
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:
        input_data = {
            "age": float(request.form["age"]),
            "sex": int(request.form["sex"]),
            "cp": int(request.form["cp"]),
            "trestbps": float(request.form["trestbps"]),
            "chol": float(request.form["chol"]),
            "fbs": int(request.form["fbs"]),
            "restecg": int(request.form["restecg"]),
            "thalach": float(request.form["thalach"]),
            "exang": int(request.form["exang"]),
            "oldpeak": float(request.form["oldpeak"]),
            "slope": int(request.form["slope"]),
            "ca": int(request.form["ca"]),
            "thal": int(request.form["thal"])
        }

        result = prediction_pipeline.predict(input_data)

        return render_template(
            "index.html",
            prediction=result["result"]
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid values for all fields."
        )


if __name__ == "__main__":
    app.run(debug=True)
