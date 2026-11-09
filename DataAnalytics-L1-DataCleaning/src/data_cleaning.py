"""Reusable cleaning pipeline for the synthetic customer transactions dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "original_dataset.csv"
CLEAN_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
CHART_DIR = PROJECT_ROOT / "visualizations"
NUMERIC_COLUMNS = ["age", "quantity", "unit_price", "total_amount"]
CATEGORICAL_COLUMNS = ["gender", "city", "product_category", "payment_method"]
GENDER_MAP = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
CITY_MAP = {"Ny": "New York", "Nyc": "New York"}
CATEGORY_MAP = {}
PAYMENT_MAP = {"Cc": "Credit Card"}


def count_outliers(data, columns=NUMERIC_COLUMNS):
    """Count IQR outliers in each available numeric column."""
    counts = {}
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            counts[column] = 0
            continue
        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        iqr = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * iqr
        upper_bound = third_quartile + 1.5 * iqr
        counts[column] = int(((values < lower_bound) | (values > upper_bound)).sum())
    return counts


def count_data_type_issues(data):
    """Count columns that do not use the project’s expected data types."""
    expected_numeric = ["age", "quantity", "unit_price", "total_amount"]
    issues = sum(not pd.api.types.is_numeric_dtype(data[column]) for column in expected_numeric)
    if "purchase_date" in data and not pd.api.types.is_datetime64_any_dtype(data["purchase_date"]):
        issues += 1
    return int(issues)


def build_quality_report(data):
    """Build a row-based report of measurable quality issues."""
    rows = [
        {"Check": "Dataset size", "Column": "All", "Count": len(data), "Details": "Rows in the dataset"},
        {"Check": "Column count", "Column": "All", "Count": len(data.columns), "Details": "Columns in the dataset"},
        {
            "Check": "Exact duplicate rows",
            "Column": "All",
            "Count": int(data.duplicated().sum()),
            "Details": "Rows that exactly match an earlier record",
        },
        {
            "Check": "Incorrect data type columns",
            "Column": "All",
            "Count": count_data_type_issues(data),
            "Details": "Expected numeric fields and purchase_date are checked",
        },
    ]

    for column in data.columns:
        missing_count = int(data[column].isna().sum())
        rows.append(
            {
                "Check": "Missing values",
                "Column": column,
                "Count": missing_count,
                "Details": f"{missing_count / len(data) * 100:.2f}% of rows",
            }
        )

    for column in ["gender", "city", "product_category", "payment_method"]:
        values = data[column].dropna().astype(str)
        if column == "gender":
            standardized = values.str.strip().str.lower().map(GENDER_MAP)
        else:
            standardized = values.str.strip().str.title()
            if column == "city":
                standardized = standardized.replace(CITY_MAP)
            elif column == "product_category":
                standardized = standardized.replace(CATEGORY_MAP)
            elif column == "payment_method":
                standardized = standardized.replace(PAYMENT_MAP)
        inconsistent_count = int((values != standardized).sum())
        rows.append(
            {
                "Check": "Inconsistent text formatting",
                "Column": column,
                "Count": inconsistent_count,
                "Details": "Values differing from the documented standardized form",
            }
        )

    quantity = pd.to_numeric(data["quantity"], errors="coerce")
    unit_price = pd.to_numeric(data["unit_price"], errors="coerce")
    total_amount = pd.to_numeric(data["total_amount"], errors="coerce")
    complete_amounts = quantity.notna() & unit_price.notna() & total_amount.notna()
    mismatch_count = int(
        (~np.isclose(
            total_amount[complete_amounts],
            (quantity[complete_amounts] * unit_price[complete_amounts]).round(2),
            atol=0.01,
        )).sum()
    )
    rows.append(
        {
            "Check": "Transaction total mismatch",
            "Column": "total_amount",
            "Count": mismatch_count,
            "Details": "Expected quantity multiplied by unit price",
        }
    )

    for column in ["purchase_date"]:
        if not pd.api.types.is_datetime64_any_dtype(data[column]):
            invalid_count = int(
                pd.to_datetime(data[column], errors="coerce", format="mixed").isna().sum()
            )
        else:
            invalid_count = int(data[column].isna().sum())
        rows.append(
            {
                "Check": "Invalid or unparseable values",
                "Column": column,
                "Count": invalid_count,
                "Details": "Date strings that cannot be parsed",
            }
        )

    for column in ["age", "quantity", "unit_price"]:
        values = pd.to_numeric(data[column], errors="coerce")
        if column == "age":
            invalid_count = int(((values <= 0) | (values > 100)).sum())
            description = "Age must be between 1 and 100"
        else:
            invalid_count = int((values <= 0).sum())
            description = "Value must be greater than zero"
        rows.append(
            {
                "Check": "Invalid or suspicious values",
                "Column": column,
                "Count": invalid_count,
                "Details": description,
            }
        )

    for column, count in count_outliers(data).items():
        rows.append(
            {
                "Check": "IQR outliers",
                "Column": column,
                "Count": count,
                "Details": "Potential outliers; review context before changing valid values",
            }
        )

    return pd.DataFrame(rows)


def clean_dataset(raw_data):
    """Return a cleaned copy and a record of the changes made."""
    cleaned = raw_data.copy()
    changes = {
        "duplicate_rows_removed": int(cleaned.duplicated().sum()),
        "invalid_date_rows_removed": 0,
        "imputed_values": {},
        "invalid_values_marked_missing": {},
    }

    cleaned = cleaned.drop_duplicates().copy()

    # Trim text first, then map only clear spelling/case variants.
    for column in ["customer_id", *CATEGORICAL_COLUMNS]:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace("", pd.NA)

    cleaned["gender"] = cleaned["gender"].str.lower().map(GENDER_MAP)
    cleaned["city"] = cleaned["city"].str.title().replace(CITY_MAP)
    cleaned["product_category"] = cleaned["product_category"].str.title().replace(CATEGORY_MAP)
    cleaned["payment_method"] = cleaned["payment_method"].str.title().replace(PAYMENT_MAP)

    cleaned["purchase_date"] = pd.to_datetime(
        cleaned["purchase_date"], errors="coerce", format="mixed"
    )
    changes["invalid_date_rows_removed"] = int(cleaned["purchase_date"].isna().sum())
    cleaned = cleaned.dropna(subset=["purchase_date"]).copy()

    for column in ["age", "quantity", "unit_price", "total_amount"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    invalid_rules = {
        "age": (cleaned["age"] <= 0) | (cleaned["age"] > 100),
        "quantity": cleaned["quantity"] <= 0,
        "unit_price": cleaned["unit_price"] <= 0,
    }
    for column, invalid_mask in invalid_rules.items():
        changes["invalid_values_marked_missing"][column] = int(invalid_mask.sum())
        cleaned.loc[invalid_mask, column] = np.nan

    # Medians are robust to skew and extreme but valid purchases.
    for column in ["age", "quantity", "unit_price"]:
        missing_count = int(cleaned[column].isna().sum())
        median_value = cleaned[column].median()
        if column == "quantity":
            median_value = int(round(median_value))
        cleaned[column] = cleaned[column].fillna(median_value)
        changes["imputed_values"][column] = missing_count

    cleaned["age"] = cleaned["age"].round().astype("Int64")
    cleaned["quantity"] = cleaned["quantity"].round().astype("Int64")

    for column in CATEGORICAL_COLUMNS:
        missing_count = int(cleaned[column].isna().sum())
        mode_values = cleaned[column].mode(dropna=True)
        if not mode_values.empty:
            cleaned[column] = cleaned[column].fillna(mode_values.iloc[0])
        changes["imputed_values"][column] = missing_count

    # The transaction amount is defined by the business rule quantity * unit price.
    cleaned["total_amount"] = cleaned["quantity"] * cleaned["unit_price"]

    cleaned["customer_id"] = cleaned["customer_id"].astype("string")
    cleaned["gender"] = cleaned["gender"].astype("string")
    cleaned["city"] = cleaned["city"].astype("string")
    cleaned["product_category"] = cleaned["product_category"].astype("string")
    cleaned["payment_method"] = cleaned["payment_method"].astype("string")
    cleaned["unit_price"] = cleaned["unit_price"].astype(float).round(2)
    cleaned["total_amount"] = cleaned["total_amount"].astype(float).round(2)
    cleaned["age"] = cleaned["age"].astype("Int64")
    cleaned["quantity"] = cleaned["quantity"].astype("Int64")

    return cleaned.reset_index(drop=True), changes


def create_before_visualizations(raw_data):
    """Save charts that diagnose the raw dataset before cleaning."""
    sns.set_theme(style="whitegrid", palette="deep")
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    missing = raw_data.isna().sum().sort_values(ascending=False)
    fig, axis = plt.subplots(figsize=(10, 5))
    missing[missing > 0].plot(kind="bar", ax=axis, color="#4472C4")
    axis.set(title="Missing Values by Column (Before Cleaning)", ylabel="Missing cells", xlabel="")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "missing_values.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5))
    sns.heatmap(raw_data.isna(), cbar=False, yticklabels=False, cmap="mako", ax=axis)
    axis.set(title="Missing-Value Pattern (Before Cleaning)", xlabel="Columns", ylabel="Records")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "missing_values_heatmap.png", dpi=150)
    plt.close(fig)

    duplicate_count = int(raw_data.duplicated().sum())
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Unique rows", "Exact duplicates"], [len(raw_data) - duplicate_count, duplicate_count],
             color=["#70AD47", "#ED7D31"])
    axis.set(title="Exact Duplicate Record Summary", ylabel="Rows")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "duplicate_analysis.png", dpi=150)
    plt.close(fig)

    numeric_before = raw_data[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for column, axis in zip(NUMERIC_COLUMNS, axes.flat):
        numeric_before[column].dropna().hist(ax=axis, bins=15, color="#5B9BD5", edgecolor="white")
        axis.set(title=f"{column.replace('_', ' ').title()} Distribution", xlabel=column, ylabel="Frequency")
    fig.suptitle("Numerical Distributions Before Cleaning")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "numerical_distributions_before.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=numeric_before, orient="h", ax=axis, color="#70AD47")
    axis.set(title="IQR Review of Numeric Fields Before Cleaning", xlabel="Value")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "outlier_before.png", dpi=150)
    plt.close(fig)


def create_after_visualizations(raw_data, clean_data):
    """Save outlier diagnostics for the cleaned numeric fields."""
    sns.set_theme(style="whitegrid", palette="deep")
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    numeric_clean = clean_data[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=numeric_clean, orient="h", ax=axis, color="#70AD47")
    axis.set(title="IQR Review of Numeric Fields After Cleaning", xlabel="Value")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "outlier_after.png", dpi=150)
    plt.close(fig)


def create_comparison_visualization(raw_data, clean_data):
    """Save the summary chart of data-quality checks before and after cleaning."""
    sns.set_theme(style="whitegrid", palette="deep")
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    duplicate_count = int(raw_data.duplicated().sum())
    raw_metrics = {
        "Missing values": int(raw_data.isna().sum().sum()),
        "Duplicate rows": duplicate_count,
        "Invalid dates": int(
            pd.to_datetime(
                raw_data["purchase_date"], errors="coerce", format="mixed"
            ).isna().sum()
        ),
    }
    clean_metrics = {
        "Missing values": int(clean_data.isna().sum().sum()),
        "Duplicate rows": int(clean_data.duplicated().sum()),
        "Invalid dates": int(clean_data["purchase_date"].isna().sum()),
    }
    comparison = pd.DataFrame(
        {"Before cleaning": raw_metrics, "After cleaning": clean_metrics}
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    comparison.plot(kind="bar", ax=axis, color=["#ED7D31", "#70AD47"])
    axis.set(title="Data Quality Checks Before and After Cleaning", ylabel="Count", xlabel="")
    axis.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "data_quality_comparison.png", dpi=150)
    plt.close(fig)


def create_visualizations(raw_data, clean_data):
    """Save the complete reproducible set of before-and-after charts."""
    create_before_visualizations(raw_data)
    create_after_visualizations(raw_data, clean_data)
    create_comparison_visualization(raw_data, clean_data)


def build_comparison(raw_data, clean_data):
    """Create the requested before-and-after metrics using observed data."""
    metrics = [
        ("Total rows", len(raw_data), len(clean_data)),
        ("Total columns", len(raw_data.columns), len(clean_data.columns)),
        ("Missing values", int(raw_data.isna().sum().sum()), int(clean_data.isna().sum().sum())),
        ("Duplicate rows", int(raw_data.duplicated().sum()), int(clean_data.duplicated().sum())),
        ("Data type issues", count_data_type_issues(raw_data), count_data_type_issues(clean_data)),
        ("IQR outliers", sum(count_outliers(raw_data).values()), sum(count_outliers(clean_data).values())),
    ]
    return pd.DataFrame(metrics, columns=["Metric", "Before Cleaning", "After Cleaning"])


def run_pipeline():
    """Run the full pipeline and write all requested deliverables."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset was not found: {RAW_FILE}")

    raw_data = pd.read_csv(RAW_FILE)
    clean_data, changes = clean_dataset(raw_data)

    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw_quality = build_quality_report(raw_data)
    clean_quality = build_quality_report(clean_data)
    comparison = build_comparison(raw_data, clean_data)

    clean_data.to_csv(CLEAN_FILE, index=False)
    raw_quality.to_csv(REPORT_DIR / "data_quality_report.csv", index=False)
    comparison.to_csv(REPORT_DIR / "before_after_comparison.csv", index=False)
    clean_quality.to_csv(REPORT_DIR / "cleaned_data_quality_report.csv", index=False)
    create_visualizations(raw_data, clean_data)

    print(f"Raw rows: {len(raw_data)}")
    print(f"Cleaned rows: {len(clean_data)}")
    print(f"Exact duplicates removed: {changes['duplicate_rows_removed']}")
    print(f"Rows removed for invalid dates: {changes['invalid_date_rows_removed']}")
    print(f"Cleaned dataset: {CLEAN_FILE}")
    print(f"Reports: {REPORT_DIR}")
    print(f"Visualizations: {CHART_DIR}")
    return raw_data, clean_data, changes


if __name__ == "__main__":
    run_pipeline()
