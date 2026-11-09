# Wine Quality Prediction: Key Findings

These findings are generated from the downloaded UCI red wine dataset and the held-out test results.

- **Data cleaning:** 1,599 rows loaded; 240 duplicate rows removed; 1,359 rows retained.
- **Most common class:** Low Quality (640 wines; 47.1% of cleaned records).
- **Class distribution:** Low Quality: 640, Medium Quality: 535, High Quality: 184.
- **Feature correlations with original quality:** strongest by absolute linear correlation are alcohol (0.480) and volatile acidity (-0.395). Correlation is descriptive and does not imply causation.
- **Best model by macro F1-score:** Random Forest (accuracy 0.610; macro precision 0.603; macro recall 0.598; macro F1 0.600).
- **Class imbalance:** the three grouped classes are not equally represented; macro-averaged metrics are reported to give each class equal weight.
- **Limitations:** the data covers Portuguese red wines only; the grouped class boundaries simplify an ordinal quality score; validation on additional wine sources is needed before practical use.

## Model comparison

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
| --- | --- | --- | --- | --- |
| Random Forest | 0.6103 | 0.6027 | 0.5980 | 0.5999 |
| SVC | 0.5919 | 0.5649 | 0.6098 | 0.5804 |
| SGD | 0.4632 | 0.4519 | 0.4987 | 0.4428 |

## Random Forest feature importance

- alcohol: 0.1765
- sulphates: 0.1255
- volatile acidity: 0.1054
- density: 0.0876
- total sulfur dioxide: 0.0855
