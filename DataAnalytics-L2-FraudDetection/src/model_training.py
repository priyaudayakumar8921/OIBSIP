"""Train reproducible SMOTE-based fraud classification pipelines."""

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def make_model_pipelines():
    """Build end-to-end pipelines so scaling and SMOTE fit on training data only."""
    logistic_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    forest_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=16,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return {
        "Logistic Regression": logistic_pipeline,
        "Random Forest": forest_pipeline,
    }


def get_smote_counts(model, y_train):
    """Return original and resampled training counts from a fitted pipeline."""
    before = y_train.value_counts().sort_index()
    additions = model.named_steps["smote"].sampling_strategy_
    after = before.copy()
    for label, added_count in additions.items():
        after.loc[label] += int(added_count)
    return pd.DataFrame(
        {
            "Class": [0, 1],
            "Before SMOTE": [int(before.get(0, 0)), int(before.get(1, 0))],
            "After SMOTE": [int(after.get(0, 0)), int(after.get(1, 0))],
        }
    )


def train_models(x_train, y_train):
    """Fit both imbalanced-learn pipelines and save them for later inference."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pipelines = make_model_pipelines()
    trained_models = {}

    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(x_train, y_train)
        trained_models[name] = pipeline
        filename = (
            "logistic_regression_model.joblib"
            if name == "Logistic Regression"
            else "random_forest_model.joblib"
        )
        joblib.dump(pipeline, MODEL_DIR / filename, compress=3)
        print(f"Saved {name} pipeline to {MODEL_DIR / filename}")

    smote_counts = get_smote_counts(trained_models["Logistic Regression"], y_train)
    if trained_models["Logistic Regression"].named_steps["smote"].sampling_strategy_ != (
        trained_models["Random Forest"].named_steps["smote"].sampling_strategy_
    ):
        raise RuntimeError("The models used inconsistent SMOTE training distributions.")

    return trained_models, smote_counts


if __name__ == "__main__":
    from data_preprocessing import load_dataset, split_features_target

    data = load_dataset()
    x_train, x_test, y_train, y_test = split_features_target(data)
    models, counts = train_models(x_train, y_train)
    print("SMOTE training counts:")
    print(counts.to_string(index=False))
    print(f"Untouched test rows: {len(x_test)}; fraud cases: {int(y_test.sum())}")
