
import joblib
import pandas as pd


class PredictionPipeline:

    def __init__(self, model_path="models/logistic_regression_optimized.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, input_data):

        # Convert user input into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Generate prediction
        prediction = self.model.predict(input_df)[0]

        # Return readable result
        if prediction == 1:
            return {
                "prediction": 1,
                "result": "Heart Disease Risk Detected"
            }

        return {
            "prediction": 0,
            "result": "No Heart Disease Risk Detected"
        }
