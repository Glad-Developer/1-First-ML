import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline, Inputed_data

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("prediction.html")
    else:
        inputed_data = Inputed_data(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        df_inputed_data = inputed_data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(df_inputed_data)

        return render_template("prediction.html", results=f"{prediction[0]:.2f}")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
