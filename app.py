"""Flask application for the Emotion Detection Using Linear SVM project."""

from functools import wraps
import json

import joblib
from flask import Flask, flash, redirect, render_template, request, session, url_for
from mysql.connector import Error

from config import ADMIN_PASSWORD, ADMIN_USERNAME, LABEL_MAP, METRICS_PATH, MODEL_PATH, SECRET_KEY
from database import (
    clear_prediction_history as delete_prediction_history,
    database_connection,
    get_prediction_history,
    insert_prediction,
)
from dataset_utils import DatasetError, inspect_dataset, readable_label


app = Flask(__name__)
app.config.update(
    SECRET_KEY=SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

try:
    model = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None
    model_error = None if model is not None else "Trained model not found. Run `python train_model.py` first."
except Exception as exc:
    model = None
    model_error = f"The trained model could not be loaded: {exc}"

try:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.is_file() else None
except (OSError, json.JSONDecodeError) as exc:
    metrics = None
    metrics_error = f"Evaluation metrics could not be read: {exc}"
else:
    metrics_error = None


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("index"))
        return view(*args, **kwargs)
    return wrapped


def dashboard_data():
    counts = {}
    classes = []
    errors = []
    from config import DATASET_DIR, DATASET_FILES
    for split, filename in DATASET_FILES.items():
        try:
            details = inspect_dataset(DATASET_DIR / filename, split, LABEL_MAP)
            counts[split] = details["records"]
            if split == "training":
                classes = details["classes"]
        except DatasetError as exc:
            counts[split] = None
            errors.append(str(exc))
    return counts, classes, errors


@app.route("/", methods=["GET"])
def index():
    if session.get("authenticated"):
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session.clear()
        session["authenticated"] = True
        session["username"] = username
        return redirect(url_for("dashboard"))
    flash("The username or password is incorrect.", "danger")
    return redirect(url_for("index"))


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    counts, classes, errors = dashboard_data()
    testing_accuracy = metrics.get("testing", {}).get("accuracy") if metrics else None
    return render_template("dashboard.html", counts=counts, classes=classes,
                           dataset_errors=errors, testing_accuracy=testing_accuracy,
                           model_error=model_error)


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    prediction = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("text", "").strip()
        if not input_text:
            flash("Enter a sentence or short paragraph before predicting.", "warning")
        elif len(input_text) > 10000:
            flash("Please keep the input under 10,000 characters.", "warning")
        elif model is None:
            flash(model_error or "The model is unavailable. Train it before predicting.", "danger")
        else:
            try:
                prediction = readable_label(model.predict([input_text])[0], LABEL_MAP)
                try:
                    insert_prediction(input_text, prediction)
                except Error as exc:
                    flash(f"Prediction succeeded, but MySQL history could not be saved: {exc}", "warning")
            except Exception as exc:
                flash(f"Prediction failed: {exc}", "danger")
    return render_template("prediction.html", prediction=prediction, input_text=input_text,
                           model_error=model_error)


@app.route("/dataset-info")
@login_required
def dataset_info():
    from config import DATASET_DIR, DATASET_FILES
    datasets = []
    errors = []
    for split, filename in DATASET_FILES.items():
        try:
            datasets.append(inspect_dataset(DATASET_DIR / filename, split, LABEL_MAP))
        except DatasetError as exc:
            errors.append(str(exc))
    return render_template("dataset_info.html", datasets=datasets, dataset_errors=errors)


@app.route("/model-performance")
@login_required
def model_performance():
    return render_template("model_performance.html", metrics=metrics,
                           metrics_error=metrics_error, model_error=model_error)


@app.route("/prediction-history")
@login_required
def prediction_history():
    records = []
    database_error = None
    try:
        records = get_prediction_history()
    except Error as exc:
        database_error = f"Could not load prediction history: {exc}. Run `python init_db.py` and check MySQL settings."
    return render_template("prediction_history.html", records=records, database_error=database_error)


@app.route("/prediction-history/clear", methods=["POST"])
@login_required
def clear_prediction_history():
    try:
        delete_prediction_history()
        flash("Prediction history has been cleared.", "success")
    except Error as exc:
        flash(f"Could not clear prediction history: {exc}", "danger")
    return redirect(url_for("prediction_history"))


if __name__ == "__main__":
    app.run(debug=True)