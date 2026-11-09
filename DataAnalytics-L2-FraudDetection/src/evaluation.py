"""Evaluate fraud models and save reports and diagnostic visualizations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
CHART_DIR = PROJECT_ROOT / "visualizations"


def evaluate_models(models, x_test, y_test):
    """Calculate metrics on the original, imbalanced test data."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    rows = []
    reports = []
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        predicted = model.predict(x_test)
        probability = model.predict_proba(x_test)[:, 1]
        predictions[name] = predicted
        probabilities[name] = probability

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision": precision_score(y_test, predicted, zero_division=0),
                "Recall": recall_score(y_test, predicted, zero_division=0),
                "F1-score": f1_score(y_test, predicted, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probability),
            }
        )
        reports.append(
            f"=== {name} ===\n"
            + classification_report(
                y_test,
                predicted,
                labels=[0, 1],
                target_names=["Legitimate (0)", "Fraud (1)"],
                zero_division=0,
                digits=4,
            )
        )

        matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
        fig, axis = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=axis,
        )
        axis.set(
            title=f"Confusion Matrix — {name}",
            xlabel="Predicted class",
            ylabel="Actual class",
        )
        fig.tight_layout()
        filename = (
            "confusion_matrix_logistic.png"
            if name == "Logistic Regression"
            else "confusion_matrix_random_forest.png"
        )
        fig.savefig(CHART_DIR / filename, dpi=160)
        plt.close(fig)

    comparison = pd.DataFrame(rows).set_index("Model")
    comparison.to_csv(REPORT_DIR / "model_comparison.csv")
    (REPORT_DIR / "classification_reports.txt").write_text(
        "\n".join(reports), encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(8, 6))
    for name, probability in probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probability)
        area = auc(false_positive_rate, true_positive_rate)
        axis.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC = {area:.4f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    axis.set(
        title="Receiver Operating Characteristic",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for name, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probability)
        axis.plot(recall, precision, label=name)
    baseline = float(np.mean(y_test))
    axis.axhline(baseline, linestyle="--", color="gray", label=f"Fraud prevalence = {baseline:.4f}")
    axis.set(
        title="Precision–Recall Curves",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    return comparison, predictions, probabilities


def save_feature_importance(model, feature_names):
    """Export the Random Forest feature importances and top-ten chart."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    forest = model.named_steps["classifier"]
    importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": forest.feature_importances_}
    ).sort_values("Importance", ascending=False)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    top_features = importance.head(10).sort_values("Importance")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top_features["Feature"], top_features["Importance"], color="#4472C4")
    axis.set(title="Random Forest — Top 10 Feature Importances", xlabel="Importance", ylabel="")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)
    return importance


def save_error_analysis(x_test, y_test, predictions, probabilities, model_name):
    """Save up to five false positives and false negatives for one model."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if model_name not in predictions or model_name not in probabilities:
        raise ValueError(f"No predictions were supplied for {model_name!r}.")

    predicted = np.asarray(predictions[model_name])
    actual = np.asarray(y_test)
    probabilities_for_fraud = np.asarray(probabilities[model_name])
    false_positive = np.flatnonzero((actual == 0) & (predicted == 1))[:5]
    false_negative = np.flatnonzero((actual == 1) & (predicted == 0))[:5]
    selected = np.concatenate([false_positive, false_negative])
    examples = pd.DataFrame(
        {
            "Test_Row_Position": selected,
            "Error_Type": [
                "False Positive" if actual[position] == 0 else "False Negative"
                for position in selected
            ],
            "Actual_Class": actual[selected],
            "Predicted_Class": predicted[selected],
            "Predicted_Fraud_Probability": probabilities_for_fraud[selected],
        }
    )
    examples.insert(0, "Model", model_name)
    examples.to_csv(REPORT_DIR / "misclassified_examples.csv", index=False)
    return examples


if __name__ == "__main__":
    from data_preprocessing import load_dataset, split_features_target

    data = load_dataset()
    x_train, x_test, y_train, y_test = split_features_target(data)
    logistic_path = PROJECT_ROOT / "models" / "logistic_regression_model.joblib"
    forest_path = PROJECT_ROOT / "models" / "random_forest_model.joblib"
    if not logistic_path.exists() or not forest_path.exists():
        raise FileNotFoundError(
            "Train both models first by running `python src/model_training.py`."
        )

    models = {
        "Logistic Regression": joblib.load(logistic_path),
        "Random Forest": joblib.load(forest_path),
    }
    comparison, predictions, probabilities = evaluate_models(models, x_test, y_test)
    importance = save_feature_importance(models["Random Forest"], x_train.columns)
    best_model_name = comparison.sort_values(
        ["Recall", "Precision", "F1-score"], ascending=False
    ).index[0]
    errors = save_error_analysis(
        x_test, y_test, predictions, probabilities, model_name=best_model_name
    )
    print("Model comparison:")
    print(comparison.to_string())
    print(f"Recall-first model: {best_model_name}")
    print(f"Top feature: {importance.iloc[0]['Feature']}")
    print(f"Misclassified examples saved: {len(errors)}")
