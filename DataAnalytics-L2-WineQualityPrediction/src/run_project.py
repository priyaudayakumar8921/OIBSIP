"""Run the complete wine-quality project from the command line."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import load_and_clean_data, save_cleaned_data, split_features_target
from model_evaluation import evaluate_models, save_feature_importance
from model_training import train_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_eda_charts(data, visualizations_dir):
    """Create descriptive plots from the cleaned, real dataset."""
    plots = Path(visualizations_dir)
    plots.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.countplot(data=data, x="quality", color="#4472C4", ax=axis)
    axis.set(title="Red Wine Quality Distribution", xlabel="Original quality score", ylabel="Number of wines")
    figure.tight_layout()
    figure.savefig(plots / "quality_distribution.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 8))
    sns.heatmap(data.drop(columns="quality_label").corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set_title("Correlation Between Wine Properties")
    figure.tight_layout()
    figure.savefig(plots / "correlation_heatmap.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="quality", y="alcohol", color="#70AD47", ax=axis)
    axis.set(title="Alcohol Content by Original Quality Score", xlabel="Original quality score", ylabel="Alcohol (% by volume)")
    figure.tight_layout()
    figure.savefig(plots / "alcohol_vs_quality.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, column in zip(axes, ["volatile acidity", "sulphates", "citric acid"]):
        sns.boxplot(data=data, x="quality", y=column, color="#ED7D31", ax=axis)
        axis.set(xlabel="Quality score", ylabel=column.title())
    figure.suptitle("Selected Physicochemical Properties by Quality")
    figure.tight_layout()
    figure.savefig(plots / "acidity_analysis.png", dpi=160)
    plt.close(figure)


def write_findings(data, audit, comparison, feature_importance, reports_dir):
    """Write findings directly from computed dataset and evaluation outputs."""
    class_counts = data["quality_label"].value_counts()
    top_class = class_counts.idxmax()
    quality_correlations = (
        data.drop(columns="quality_label")
        .corr()["quality"]
        .drop("quality")
        .sort_values(key=abs, ascending=False)
    )
    best = comparison.iloc[0]
    findings = [
        "# Wine Quality Prediction: Key Findings",
        "",
        "These findings are generated from the downloaded UCI red wine dataset and the held-out test results.",
        "",
        f"- **Data cleaning:** {audit['rows_loaded']:,} rows loaded; {audit['duplicate_rows_removed']:,} duplicate rows removed; {len(data):,} rows retained.",
        f"- **Most common class:** {top_class} ({class_counts[top_class]:,} wines; {class_counts[top_class] / len(data):.1%} of cleaned records).",
        f"- **Class distribution:** " + ", ".join(f"{label}: {count:,}" for label, count in class_counts.reindex(["Low Quality", "Medium Quality", "High Quality"], fill_value=0).items()) + ".",
        f"- **Feature correlations with original quality:** strongest by absolute linear correlation are {quality_correlations.index[0]} ({quality_correlations.iloc[0]:.3f}) and {quality_correlations.index[1]} ({quality_correlations.iloc[1]:.3f}). Correlation is descriptive and does not imply causation.",
        f"- **Best model by macro F1-score:** {best['Model']} (accuracy {best['Accuracy']:.3f}; macro precision {best['Precision (Macro)']:.3f}; macro recall {best['Recall (Macro)']:.3f}; macro F1 {best['F1-score (Macro)']:.3f}).",
        "- **Class imbalance:** the three grouped classes are not equally represented; macro-averaged metrics are reported to give each class equal weight.",
        "- **Limitations:** the data covers Portuguese red wines only; the grouped class boundaries simplify an ordinal quality score; validation on additional wine sources is needed before practical use.",
        "",
        "## Model comparison",
        "",
        "| " + " | ".join(comparison.columns) + " |",
        "| " + " | ".join("---" for _ in comparison.columns) + " |",
    ]
    findings.extend(
        "| " + " | ".join(
            f"{value:.4f}" if isinstance(value, float) else str(value)
            for value in row
        ) + " |"
        for row in comparison.itertuples(index=False, name=None)
    )
    findings.extend([
        "",
        "## Random Forest feature importance",
        "",
    ])
    findings.extend(
        f"- {name}: {value:.4f}"
        for name, value in feature_importance.head(5).items()
    )
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    (Path(reports_dir) / "key_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")


def main():
    raw_path = PROJECT_ROOT / "data" / "raw" / "winequality-red.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            "Dataset missing. Download the UCI Wine Quality archive and place "
            "winequality-red.csv in data/raw/."
        )
    data, audit = load_and_clean_data(raw_path)
    save_cleaned_data(data, PROJECT_ROOT / "data" / "processed" / "cleaned_wine_data.csv")
    visualizations = PROJECT_ROOT / "visualizations"
    create_eda_charts(data, visualizations)
    X_train, X_test, y_train, y_test = split_features_target(data)
    models = train_models(X_train, y_train, PROJECT_ROOT / "models")
    comparison, _ = evaluate_models(
        models, X_test, y_test, PROJECT_ROOT / "reports", visualizations
    )
    importance = save_feature_importance(models["Random Forest"], X_train.columns, visualizations)
    write_findings(data, audit, comparison, importance, PROJECT_ROOT / "reports")
    print(f"Cleaned rows: {len(data)}")
    print(comparison.to_string(index=False))
    print(f"Best model by macro F1: {comparison.iloc[0]['Model']}")


if __name__ == "__main__":
    main()
