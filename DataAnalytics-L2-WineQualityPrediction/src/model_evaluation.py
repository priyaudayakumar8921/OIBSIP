"""Evaluate classifiers and create reusable reports and visualizations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_preprocessing import QUALITY_LABELS


def evaluate_models(models, X_test, y_test, reports_dir, visualizations_dir):
    """Evaluate every model with macro metrics and save detailed artifacts."""
    reports_path = Path(reports_dir)
    plots_path = Path(visualizations_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    rows = []
    report_text = []
    predictions = {}
    sns.set_theme(style="whitegrid")

    for name, model in models.items():
        predicted = model.predict(X_test)
        predictions[name] = predicted
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision (Macro)": precision_score(
                    y_test, predicted, average="macro", zero_division=0
                ),
                "Recall (Macro)": recall_score(
                    y_test, predicted, average="macro", zero_division=0
                ),
                "F1-score (Macro)": f1_score(
                    y_test, predicted, average="macro", zero_division=0
                ),
            }
        )
        report_text.append(
            f"{name}\n{'=' * len(name)}\n"
            + classification_report(
                y_test, predicted, labels=QUALITY_LABELS, zero_division=0
            )
        )
        matrix = confusion_matrix(y_test, predicted, labels=QUALITY_LABELS)
        figure, axis = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=QUALITY_LABELS,
            yticklabels=QUALITY_LABELS,
            ax=axis,
        )
        axis.set(
            title=f"{name} Confusion Matrix",
            xlabel="Predicted class",
            ylabel="Actual class",
        )
        figure.tight_layout()
        filename = {
            "Random Forest": "random_forest_confusion_matrix.png",
            "SVC": "svm_confusion_matrix.png",
            "SGD": "sgd_confusion_matrix.png",
        }[name]
        figure.savefig(plots_path / filename, dpi=160)
        plt.close(figure)

    comparison = pd.DataFrame(rows).sort_values(
        "F1-score (Macro)", ascending=False
    )
    comparison.to_csv(reports_path / "model_comparison.csv", index=False)
    (reports_path / "classification_reports.txt").write_text(
        "\n\n".join(report_text), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    comparison.set_index("Model")["Accuracy"].plot(
        kind="bar", color=sns.color_palette("deep", len(comparison)), ax=axis
    )
    axis.set(
        title="Wine Quality Classifier Accuracy",
        xlabel="Model",
        ylabel="Accuracy",
        ylim=(0, 1),
    )
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    figure.savefig(plots_path / "model_accuracy_comparison.png", dpi=160)
    plt.close(figure)
    return comparison, predictions


def save_feature_importance(model, feature_names, visualizations_dir):
    """Save a ranked feature importance chart for the trained Random Forest."""
    classifier = model.named_steps["classifier"]
    importance = pd.Series(
        classifier.feature_importances_, index=feature_names
    ).sort_values()
    figure, axis = plt.subplots(figsize=(9, 6))
    importance.plot(kind="barh", color=sns.color_palette("viridis", len(importance)), ax=axis)
    axis.set(
        title="Random Forest Feature Importance",
        xlabel="Importance",
        ylabel="Physicochemical property",
    )
    figure.tight_layout()
    figure.savefig(
        Path(visualizations_dir) / "feature_importance.png", dpi=160
    )
    plt.close(figure)
    return importance.sort_values(ascending=False)
