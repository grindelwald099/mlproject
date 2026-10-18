
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app=application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    try:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get(
                "test_preparation_course"
            ),
            reading_score=request.form.get("reading_score"),
            writing_score=request.form.get("writing_score")
        )

        pred_df = data.get_data_as_frame()

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)

        result = round(float(pred[0]), 2)

        return render_template("home.html", results=result)

    except Exception as e:
        app.logger.exception("Prediction failed")
        return render_template(
            "home.html",
            error=f"Prediction failed: {str(e)}"
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0")
