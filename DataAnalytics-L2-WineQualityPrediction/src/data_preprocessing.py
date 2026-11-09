"""Load, clean, label, and split the UCI red wine quality data."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


QUALITY_LABELS = ["Low Quality", "Medium Quality", "High Quality"]
TARGET_COLUMN = "quality_label"


def load_and_clean_data(dataset_path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the semicolon-delimited UCI file and apply conservative cleaning."""
    data = pd.read_csv(dataset_path, sep=";")
    if "quality" not in data.columns:
        raise ValueError("The dataset must contain a 'quality' column.")

    data = data.apply(pd.to_numeric, errors="raise")
    if data["quality"].isna().any():
        raise ValueError("The quality target contains missing values.")

    audit = {
        "rows_loaded": len(data),
        "duplicate_rows_removed": int(data.duplicated().sum()),
        "missing_values_before_cleaning": int(data.isna().sum().sum()),
    }
    data = data.drop_duplicates().copy()

    feature_columns = data.columns.drop("quality")
    missing_features = data[feature_columns].isna().sum()
    for column in feature_columns[missing_features.gt(0)]:
        data[column] = data[column].fillna(data[column].median())

    data[TARGET_COLUMN] = data["quality"].map(
        lambda value: "Low Quality"
        if value <= 5
        else "Medium Quality"
        if value == 6
        else "High Quality"
    )
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Unable to map one or more quality values to a class.")

    audit["rows_after_cleaning"] = len(data)
    audit["missing_values_after_cleaning"] = int(data.isna().sum().sum())
    return data, audit


def save_cleaned_data(data: pd.DataFrame, output_path: str | Path) -> None:
    """Save the cleaned dataset, retaining both original and grouped targets."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def split_features_target(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create one reproducible stratified split shared by all classifiers."""
    features = data.drop(columns=["quality", TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
