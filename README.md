# Immunotherapy Adverse-Event Classification

Small ML project to predict severe adverse events from an immunotherapy cohort. Two models are benchmarked: a regularized Logistic Regression baseline and a Random Forest as a non-linear comparator. Work is done in a Jupyter notebook (`src/main.ipynb`).

## Data
- Source CSVs in `data/`
	- `immunotherapy_classification.csv` (classification target `Severe_Adverse_Event`)
- Basic preprocessing: drop ID-like columns, impute missing values (median for numeric, most-frequent for categorical), one-hot encode categoricals, drop near-constant features.

## Environment
- Python 3.11+ recommended.
- Install deps: `pip install -r requirements.txt` (scikit-learn, pandas, seaborn, matplotlib, numpy).

## How to run
- Notebook (recommended):
	1) Open `src/main.ipynb` in VS Code/Jupyter.
	2) Run all cells; figures (EDA, confusion matrices, ROC) render inline.
- Script (quick demo): `python src/main.py` (expects `data/` relative to repo root).

## Project structure
- `src/main.ipynb` — full workflow: EDA, preprocessing, Logistic Regression, Random Forest, cross-val, interpretation.
- `src/main.py` — lightweight example loading the classification CSV and fitting Logistic Regression.
- `data/` — input CSVs.
- `requirements.txt` — dependencies.

## Key steps in notebook
- EDA: target balance, missing values, numeric distributions.
- Preprocessing: imputation, one-hot encoding, scaling for linear model, train/test split (stratified).
- Models: Logistic Regression (scaled), Random Forest (`class_weight="balanced"`), ROC-AUC comparison via 5-fold CV.
- Evaluation: accuracy, ROC AUC, confusion matrices, ROC curves; coefficient and feature-importance plots.

## Typical results (classification)
- ROC AUC (5-fold): Logistic Regression ≈0.89, Random Forest ≈0.89 on this dataset; LR chosen for simplicity and stability.

