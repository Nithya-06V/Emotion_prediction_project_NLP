"""Shared CSV discovery, validation, and dataset inspection helpers."""

from pathlib import Path
import re

import pandas as pd


TEXT_CANDIDATES = (
    "text", "sentence", "content", "utterance", "message", "review",
    "tweet", "input", "text_data", "comment",
)
LABEL_CANDIDATES = (
    "label", "emotion", "sentiment", "category", "class", "target",
    "emotion_label", "sentiment_label", "feeling",
)


class DatasetError(ValueError):
    """Raised when a dataset cannot be safely interpreted."""


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def detect_columns(frame, file_name="dataset"):
    """Find text/label columns by common names, then conservative heuristics."""
    columns = list(frame.columns)
    normalized = {normalize_column_name(column): column for column in columns}
    text_column = next((normalized[name] for name in TEXT_CANDIDATES if name in normalized), None)
    label_column = next((normalized[name] for name in LABEL_CANDIDATES if name in normalized), None)

    if text_column is None:
        string_columns = [column for column in columns if pd.api.types.is_string_dtype(frame[column])
                          or frame[column].dtype == object]
        if string_columns:
            text_column = max(
                string_columns,
                key=lambda column: frame[column].fillna("").astype(str).str.len().mean(),
            )

    if label_column is None:
        candidates = [column for column in columns if column != text_column]
        candidates = [column for column in candidates if frame[column].nunique(dropna=True) <= max(50, int(len(frame) * 0.2))]
        if candidates:
            label_column = min(candidates, key=lambda column: frame[column].nunique(dropna=True))

    if text_column is None or label_column is None or text_column == label_column:
        available = ", ".join(map(str, columns)) or "(no columns found)"
        raise DatasetError(
            f"Could not identify the text and emotion columns in {file_name}. "
            f"Available columns: {available}. Rename columns to a common text name "
            "(such as 'text') and label name (such as 'label' or 'emotion'), or "
            "update TEXT_CANDIDATES/LABEL_CANDIDATES in dataset_utils.py."
        )
    return text_column, label_column


def read_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}. Place the required CSV in the dataset folder.")
    try:
        frame = pd.read_csv(path)
    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as exc:
        raise DatasetError(f"Could not read {path.name}: {exc}") from exc
    if frame.empty:
        raise DatasetError(f"{path.name} is empty.")
    text_column, label_column = detect_columns(frame, path.name)
    return frame, text_column, label_column


def clean_dataset(frame, text_column, label_column):
    cleaned = frame[[text_column, label_column]].copy()
    cleaned = cleaned.dropna(subset=[text_column, label_column])
    cleaned[text_column] = cleaned[text_column].astype(str).str.strip()
    cleaned = cleaned[cleaned[text_column].ne("")]
    if cleaned[label_column].dtype == object:
        cleaned[label_column] = cleaned[label_column].astype(str).str.strip()
        cleaned = cleaned[cleaned[label_column].ne("")]
    cleaned = cleaned.drop_duplicates(subset=[text_column, label_column]).reset_index(drop=True)
    if cleaned.empty:
        raise DatasetError("No usable records remain after removing missing or duplicate rows.")
    if cleaned[label_column].nunique() < 2:
        raise DatasetError("At least two emotion classes are required to train a classifier.")
    return cleaned


def readable_label(value, label_map):
    key = str(value)
    if key in label_map:
        return label_map[key]
    if isinstance(value, str) and not value.strip().lstrip("+-").isdigit():
        return value.strip().replace("_", " ").title()
    return f"Class {key}"


def inspect_dataset(path, split_name, label_map):
    frame, text_column, label_column = read_dataset(path)
    cleaned = clean_dataset(frame, text_column, label_column)
    counts = cleaned[label_column].value_counts(dropna=False)
    classes = [
        {"value": str(value), "name": readable_label(value, label_map), "count": int(count)}
        for value, count in counts.items()
    ]
    samples = cleaned.head(8).rename(columns={text_column: "Text", label_column: "Emotion"})
    samples["Emotion"] = samples["Emotion"].map(lambda value: readable_label(value, label_map))
    return {
        "split": split_name,
        "file": Path(path).name,
        "records": int(len(frame)),
        "usable_records": int(len(cleaned)),
        "columns": [str(column) for column in frame.columns],
        "text_column": str(text_column),
        "label_column": str(label_column),
        "classes": classes,
        "missing": {str(column): int(value) for column, value in frame.isna().sum().items()},
        "duplicates": int(frame.duplicated().sum()),
        "sample_columns": list(samples.columns),
        "samples": samples.fillna("").astype(str).to_dict(orient="records"),
    }