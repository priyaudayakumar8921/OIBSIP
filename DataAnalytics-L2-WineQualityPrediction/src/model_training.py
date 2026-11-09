"""Build and train the three requested wine-quality classifiers."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return classifiers with scaling where the algorithm requires it."""
    return {
        "Random Forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "SVC": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(kernel="rbf", class_weight="balanced", random_state=random_state),
                ),
            ]
        ),
        "SGD": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SGDClassifier(
                        loss="modified_huber",
                        class_weight="balanced",
                        max_iter=3000,
                        tol=1e-3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def train_models(
    X_train, y_train, models_dir: str | Path, random_state: int = 42
) -> dict[str, Pipeline]:
    """Fit all models and persist their complete prediction pipelines."""
    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = build_models(random_state=random_state)
    filenames = {
        "Random Forest": "random_forest_model.joblib",
        "SVC": "svm_model.joblib",
        "SGD": "sgd_model.joblib",
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, output_dir / filenames[name])
    return models
