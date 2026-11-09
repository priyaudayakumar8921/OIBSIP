# Wine Quality Prediction Using Machine Learning

**Oasis Infobyte (OIBSIP) | Level 2 | Task 2**

## Project overview

This project uses the UCI red wine quality dataset to explore physicochemical properties and classify wines into **Low Quality** (score <= 5), **Medium Quality** (score == 6), and **High Quality** (score >= 7). It compares Random Forest, Support Vector Classifier (SVC), and Stochastic Gradient Descent (SGD) classifiers.

The original numeric `quality` score is retained in the cleaned data. The grouped `quality_label` target is created separately so the classification mapping remains transparent.

## Dataset

- Source: [UCI Machine Learning Repository - Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- File used: `winequality-red.csv` (red wine)
- Expected local path: `data/raw/winequality-red.csv`
- The dataset is a real, publicly available Portuguese wine dataset; no synthetic records or fabricated results are used.

If the included dataset is missing, download the UCI Wine Quality dataset archive and copy `winequality-red.csv` into `data/raw/`.

## Project layout

```text
DataAnalytics-L2-WineQualityPrediction/
├── data/
│   ├── raw/winequality-red.csv
│   └── processed/cleaned_wine_data.csv
├── notebooks/Wine_Quality_Prediction.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── run_project.py
├── models/
│   ├── random_forest_model.joblib
│   ├── svm_model.joblib
│   └── sgd_model.joblib
├── reports/
│   ├── model_comparison.csv
│   ├── classification_reports.txt
│   └── key_findings.md
├── visualizations/
├── requirements.txt
└── .gitignore
```

## Setup and execution

Use Python 3.10 or later. From this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/run_project.py
```

To open and run the complete analysis notebook:

```powershell
jupyter notebook notebooks/Wine_Quality_Prediction.ipynb
```

The command-line workflow cleans the data, creates visualizations, fits and saves all three models, evaluates them, and writes reports. Each saved Joblib file includes the full preprocessing/model pipeline needed for later predictions.

## Analysis and preprocessing

The source file is semicolon-delimited. The cleaning script checks numeric types and missing values, removes exact duplicate rows, and median-imputes missing feature values if any occur. It does not impute or alter missing target labels. The processed file retains both `quality` and the derived `quality_label`.

The project examines the original score distribution, descriptive statistics, feature correlations, alcohol, volatile acidity, sulphates, and citric acid. The cleaned rows are split once with an 80:20 stratified train/test split (`random_state=42`), shared by all models. StandardScaler is fitted inside the SVC and SGD pipelines on training data only; Random Forest is trained without scaling.

## Models and evaluation

- **Random Forest Classifier** with class weighting
- **Support Vector Classifier** using an RBF kernel, scaling, and class weighting
- **SGD Classifier** using modified Huber loss, scaling, and class weighting

Accuracy is reported alongside macro precision, macro recall, and macro F1-score. Macro averages weight each of the three classes equally, making the comparison more informative when class counts differ. Confusion matrices and per-class classification reports are also saved. Model comparison values and the best model are calculated from the held-out test data and recorded in `reports/model_comparison.csv` and `reports/key_findings.md`; run the workflow to regenerate them for your environment.

Results from the included run (1,359 cleaned records; stratified 80:20 split):

| Model | Accuracy | Macro precision | Macro recall | Macro F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.6103 | 0.6027 | 0.5980 | 0.5999 |
| SVC | 0.5919 | 0.5649 | 0.6098 | 0.5804 |
| SGD | 0.4632 | 0.4519 | 0.4987 | 0.4428 |

Random Forest had the highest macro F1-score for this split. These are measured results for this dataset and configuration, not a guarantee of performance on new sources.

The Random Forest feature-importance chart describes the fitted forest's internal split importance; it should not be interpreted as causal evidence.

## Generated visualizations

Running the project creates:

1. `quality_distribution.png`
2. `correlation_heatmap.png`
3. `alcohol_vs_quality.png`
4. `acidity_analysis.png`
5. `random_forest_confusion_matrix.png`
6. `svm_confusion_matrix.png`
7. `sgd_confusion_matrix.png`
8. `model_accuracy_comparison.png`
9. `feature_importance.png`

## Limitations

The data covers Portuguese red wines and may not generalize to other regions or wine types. Grouping a subjective ordinal score into three labels loses detail and the classes are imbalanced. The test results are estimates on one stratified split, not proof of real-world performance.

## OIBSIP criteria covered

The project includes reusable preprocessing, model-training, and evaluation scripts; the requested three classifiers; a complete analysis notebook; trained model pipelines; actual metric and classification reports; required visualizations; and this setup guide. Dataset results are generated from the UCI file at execution time rather than being asserted in advance.
