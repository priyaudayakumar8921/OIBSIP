# Professional Data Cleaning and Preprocessing Using Python

**Oasis Infobyte (OIBSIP) | Data Analytics Internship | Level 1, Task 3 — Cleaning Data**

## Project overview

This project demonstrates a reproducible, beginner-friendly data-cleaning workflow using a deliberately messy **synthetic** customer transaction dataset. It identifies data-quality problems, documents decisions, produces an analysis-ready CSV, and compares measurable quality checks before and after cleaning.

The records are fictional and were created for this exercise. They are not sourced from customers, a business, or a public dataset, and the findings should not be interpreted as real-world behavior.

## Objectives

- Inspect missing, duplicated, inconsistently formatted, invalid, and unusually large values.
- Apply documented, appropriate handling rather than blindly dropping records.
- Preserve the raw input and export a separate cleaned dataset.
- Produce reports and visualizations that can be regenerated from the included files.

## Dataset

`data/raw/original_dataset.csv` contains fictional transactions and fields for customer ID, age, gender, city, product category, purchase date, quantity, unit price, transaction total, and payment method. It intentionally includes blank cells, exact duplicate rows, inconsistent text, malformed dates, impossible values, a mismatched transaction total, and plausible high-value observations.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, and Jupyter Notebook.

## Project structure

```text
DataAnalytics-L1-DataCleaning/
├── data/
│   ├── raw/original_dataset.csv
│   └── processed/cleaned_dataset.csv
├── notebooks/Data_Cleaning_Analysis.ipynb
├── reports/
│   ├── data_quality_report.csv
│   ├── cleaned_data_quality_report.csv
│   └── before_after_comparison.csv
├── visualizations/
├── src/data_cleaning.py
├── requirements.txt
└── README.md
```

## Cleaning decisions

1. Report missing cells, exact duplicates, parse failures, invalid values, type mismatches, and IQR outliers before making changes.
2. Remove exact duplicate records.
3. Trim whitespace and standardize clear gender, city, category casing, and payment-method variants without merging distinct categories or card types.
4. Parse dates consistently and exclude records with unparseable dates because they cannot be reliably placed in time-based analysis.
5. Treat ages outside 1–100 and non-positive quantities or prices as invalid, then impute numeric gaps with the median.
6. Fill missing categorical values with the column mode. No forward fill is used because these are independent transactions, not a time series.
7. Recalculate `total_amount` as `quantity × unit_price`, the transaction rule represented by this dataset.
8. Retain statistically flagged IQR outliers when they remain plausible purchases; IQR flags are review signals, not proof of errors.
9. Convert IDs/text, dates, counts, and monetary values to appropriate types.

The complete record of counts and the before/after comparison are generated in `reports/`. The script also writes `reports/cleaned_data_quality_report.csv` for validating the result.

## Visualizations

The pipeline saves missing-value counts and patterns, duplicate summaries, numeric distributions, before/after outlier boxplots, and a data-quality comparison in `visualizations/`.

## Results and findings

These results are calculated from the included CSV:

| Metric | Before cleaning | After cleaning |
|---|---:|---:|
| Rows | 54 | 49 |
| Columns | 10 | 10 |
| Missing cells | 5 | 0 |
| Exact duplicate rows | 4 | 0 |
| Expected data-type issues | 1 | 0 |
| IQR outlier flags across numeric columns | 22 | 19 |

Key findings:

- The raw file contains two impossible ages, two non-positive quantities, and one non-positive unit price. These values were treated as invalid and imputed using numeric medians.
- One date could not be parsed and its row was excluded. The other mixed-format dates were successfully parsed.
- One complete transaction had an amount inconsistent with quantity × unit price; transaction totals were recalculated using that documented rule.
- Text formatting was inconsistent in the gender, city, product-category, and payment-method fields; known variants were standardized without merging distinct product categories or payment types.
- Plausible IQR outliers remain flagged for domain review. They were not removed just because they are statistically unusual.

The outlier metric sums per-column IQR flags, so a row may contribute to more than one field's count. Reports and notebook findings are regenerated from the raw CSV when the project runs.

## Installation and use

From this project folder, create and activate a Python environment if desired, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the reusable pipeline:

```bash
python src/data_cleaning.py
```

Or launch Jupyter and open the notebook:

```bash
jupyter notebook
```

Run `notebooks/Data_Cleaning_Analysis.ipynb` from top to bottom. Paths are resolved relative to the project, and the original CSV is never overwritten.

## Conclusion

This project applies explicit, auditable cleaning decisions to a fictional business dataset and saves both the cleaned records and evidence of the transformation. Clean data supports more trustworthy analysis; it does not, by itself, guarantee that every remaining value is correct.

## Acknowledgement

Prepared for the Oasis Infobyte Data Analytics Internship, Level 1, Task 3. Thank you to Oasis Infobyte for the learning opportunity.
