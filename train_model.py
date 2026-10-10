"""Train and evaluate the TF-IDF / Linear SVM emotion classifier."""

import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import CONFUSION_MATRIX_PATH, DATASET_DIR, DATASET_FILES, LABEL_MAP, METRICS_PATH, MODEL_DIR, MODEL_PATH
from dataset_utils import clean_dataset, read_dataset, readable_label


def load_split(split_name):
    path = DATASET_DIR / DATASET_FILES[split_name]
    frame, text_column, label_column = read_dataset(path)
    cleaned = clean_dataset(frame, text_column, label_column)
    return cleaned[text_column].tolist(), cleaned[label_column].tolist(), {
        "file": path.name,
        "records": len(frame),
        "usable_records": len(cleaned),
        "text_column": str(text_column),
        "label_column": str(label_column),
    }


def evaluate(model, texts, labels, class_labels):
    predictions = model.predict(texts)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_labels, average="weighted", zero_division=0,
    )
    names = [readable_label(label, LABEL_MAP) for label in class_labels]
    report = classification_report(
        labels, predictions, labels=class_labels, target_names=names,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": report,
        "predictions": predictions,
    }


def main():
    try:
        train_texts, train_labels, train_info = load_split("training")
        validation_texts, validation_labels, validation_info = load_split("validation")
        test_texts, test_labels, test_info = load_split("testing")
    except Exception as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 1

    class_labels = sorted(set(train_labels), key=lambda value: str(value))
    if len(class_labels) < 2:
        print("Training data must contain at least two emotion classes.", file=sys.stderr)
        return 1
    unseen = sorted(set(validation_labels + test_labels) - set(class_labels), key=lambda value: str(value))
    if unseen:
        print(f"Validation/testing contain labels absent from training: {unseen}", file=sys.stderr)
        return 1

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=100000)),
        ("classifier", LinearSVC(random_state=42)),
    ])
    pipeline.fit(train_texts, train_labels)

    validation = evaluate(pipeline, validation_texts, validation_labels, class_labels)
    testing = evaluate(pipeline, test_texts, test_labels, class_labels)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "algorithm": "Linear SVM (LinearSVC)",
        "feature_extraction": "TF-IDF word unigrams and bigrams",
        "random_state": 42,
        "training": train_info,
        "validation_dataset": validation_info,
        "testing_dataset": test_info,
        "classes": [{"value": str(label), "name": readable_label(label, LABEL_MAP)} for label in class_labels],
        "label_mapping": {str(label): readable_label(label, LABEL_MAP) for label in class_labels},
        "validation": {key: value for key, value in validation.items() if key != "predictions"},
        "testing": {key: value for key, value in testing.items() if key != "predictions"},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matrix = confusion_matrix(test_labels, testing["predictions"], labels=class_labels)
    figure_size = max(7, len(class_labels) * 1.15)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size * 0.82))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    names = [readable_label(label, LABEL_MAP) for label in class_labels]
    axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names,
             yticklabels=names, xlabel="Predicted emotion", ylabel="True emotion",
             title="Held-out testing confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "#172b4d")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Training records used: {len(train_texts)} / {train_info['records']}")
    print(f"Validation accuracy: {validation['accuracy']:.4f}")
    print(f"Testing accuracy: {testing['accuracy']:.4f}")
    print("Held-out testing classification report:")
    print(classification_report(test_labels, testing["predictions"],
                                labels=class_labels,
                                target_names=[readable_label(label, LABEL_MAP) for label in class_labels],
                                zero_division=0))
    print("Held-out testing confusion matrix:")
    print(matrix)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())