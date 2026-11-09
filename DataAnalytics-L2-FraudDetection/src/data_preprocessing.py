"""Load, inspect, and split the public credit-card fraud dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "processed_transactions.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
TARGET_COLUMN = "Class"
RANDOM_STATE = 42


def load_dataset(file_path=RAW_FILE):
    """Load the original CSV and validate its expected benchmark columns."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it in data/raw/."
        )

    data = pd.read_csv(file_path)
    expected_columns = {"Time", "Amount", TARGET_COLUMN}
    missing_columns = expected_columns.difference(data.columns)
    feature_columns = [f"V{number}" for number in range(1, 29)]
    missing_columns.update(set(feature_columns).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing_columns)}")
    if not set(data[TARGET_COLUMN].dropna().unique()).issubset({0, 1}):
        raise ValueError("The Class target must contain only 0 (legitimate) and 1 (fraud).")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("The Class target contains missing values; cannot train classifiers.")
    if data.empty:
        raise ValueError("The dataset is empty.")
    return data


def build_class_report(data):
    """Calculate actual class counts and percentages."""
    counts = data[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    percentages = counts / len(data) * 100
    return pd.DataFrame(
        {
            "Class": [0, 1],
            "Label": ["Legitimate", "Fraudulent"],
            "Transaction_Count": [int(counts.loc[0]), int(counts.loc[1])],
            "Percentage": [float(percentages.loc[0]), float(percentages.loc[1])],
        }
    )


def save_data_reports(data):
    """Write the class-imbalance report and concise data-quality audit."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    class_report = build_class_report(data)
    class_report.to_csv(REPORT_DIR / "class_imbalance_report.csv", index=False)

    quality_report = pd.DataFrame(
        [
            {"Check": "Rows", "Value": len(data), "Notes": "Raw benchmark records"},
            {"Check": "Columns", "Value": len(data.columns), "Notes": "Includes Class target"},
            {
                "Check": "Missing cells",
                "Value": int(data.isna().sum().sum()),
                "Notes": "Checked before modeling",
            },
            {
                "Check": "Exact duplicate rows",
                "Value": int(data.duplicated().sum()),
                "Notes": "Audited and removed from the modeling copy before splitting to prevent train-test leakage",
            },
            {
                "Check": "Features",
                "Value": len(data.columns) - 1,
                "Notes": "All columns except Class",
            },
        ]
    )
    quality_report.to_csv(REPORT_DIR / "data_quality_report.csv", index=False)
    return class_report, quality_report


def prepare_processed_copy(data):
    """Write a modeling copy with exact duplicate records removed."""
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.drop_duplicates().to_csv(PROCESSED_FILE, index=False)
    return PROCESSED_FILE


def split_features_target(data):
    """Deduplicate before splitting, then create the stratified 80/20 holdout.

    Removing exact duplicates prevents identical transactions from appearing on
    both sides of the train-test boundary and inflating the holdout scores.
    """
    modeling_data = data.drop_duplicates()
    features = modeling_data.drop(columns=[TARGET_COLUMN])
    target = modeling_data[TARGET_COLUMN].astype("int8")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    dataset = load_dataset()
    class_summary, quality_summary = save_data_reports(dataset)
    processed_path = prepare_processed_copy(dataset)
    print(f"Dataset shape: {dataset.shape}")
    print(f"Processed copy: {processed_path}")
    print("Class distribution:")
    print(class_summary.to_string(index=False))
    print("Quality checks:")
    print(quality_summary.to_string(index=False))
