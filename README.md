# Hepatitis Survival Prediction — Machine Learning Classification



## Overview

Hepatitis is a liver condition that can range from mild to life-threatening. This project uses supervised machine learning to classify patients as likely to **survive** or **die** based on clinical indicators recorded at diagnosis, with the goal of supporting early risk assessment.

---

## Dataset

- **Source:** UCI Machine Learning Repository — [Hepatitis Dataset](https://archive.ics.uci.edu/dataset/46/hepatitis)
- **Target variable:** `Class` (survive / die)
- **Attributes:** 19 clinical and demographic features, including:
  - Demographics: `AGE`, `SEX`
  - Treatment indicators: `STEROID`, `ANTIVIRALS`
  - Symptoms: `FATIGUE`, `MALAISE`, `ANOREXIA`
  - Clinical exam findings: `LIVER BIG`, `LIVER FIRM`, `SPLEEN PALPABLE`, `SPIDERS`, `ASCITES`, `VARICES`
  - Lab values: `BILIRUBIN`, `ALK PHOSPHATE`, `SGOT`, `ALBUMIN`, `PROTIME`
  - `HISTOLOGY`

---

## Project Pipeline

### 1. Data Preprocessing
- Load raw data (`hepatitis.data`) and assign column headers.
- Replace missing value markers (`?`) with `NaN`.
- Drop rows containing missing values.
- Export intermediate cleaned datasets at each stage (`hepatitiss1.csv` → `hepatitiss2.csv`).

### 2. Feature Selection
- Apply **mutual information** (`SelectKBest` with `mutual_info_classif` from scikit-learn) to rank features by relevance to the target class.
- Select the top 16 most informative features.
- Export the reduced feature set (`hepatitiss3.csv`) and the version including the target column (`hepatitiss4.csv`).

### 3. Model Training & Classification
Multiple classification algorithms are trained and compared on the processed data:

| Model | Description |
|---|---|
| **Logistic Regression** | Baseline linear classifier for binary outcome prediction |
| **Random Forest** | Ensemble of decision trees for improved accuracy and robustness to overfitting |
| **Support Vector Machine (SVM)** | Margin-based classifier, effective for smaller, high-dimensional clinical datasets |

*(Fill in: train/test split ratio, cross-validation strategy, hyperparameters used for each model.)*

### 4. Evaluation
Models are evaluated and compared to identify the best-performing classifier for this dataset.

*(Fill in: metrics used — e.g., Accuracy, Precision, Recall, F1-score, ROC-AUC — and final results table.)*

---

## Repository Structure

```
hepatitiss/
├── hepatitis.data          # Raw dataset
├── hepatitis_1.ipynb       # Data cleaning & feature selection
├── <model_notebook>.ipynb  # Model training & evaluation (update filename)
├── hepatitiss1.csv         # Data after NaN replacement
├── hepatitiss2.csv         # Data after dropping missing rows
├── hepatitiss3.csv         # Top 16 selected features (no target)
├── hepatitiss4.csv         # Top 16 selected features + target
├── classes.csv             # Target labels only
└── README.md
```

---

## Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy

Install dependencies:

```bash
pip install pandas scikit-learn numpy
```

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/diasekar77-sketch/hepatitiss.git
   cd hepatitiss
   ```
2. Ensure `hepatitis.data` is in the project directory (update file paths in the notebooks if needed).
3. Run the preprocessing notebook (`hepatitis_1.ipynb`) to generate cleaned CSVs and selected features.
4. Run the model training notebook to train and evaluate Logistic Regression, Random Forest, and SVM classifiers.



## Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Cross-validation for more robust performance estimates
- Handling class imbalance if present in the dataset
- Model interpretability (e.g., feature importance, SHAP values)
- Deployment as a simple web app for clinical use

---

## Author

Divya Sekar — [GitHub](https://github.com/diasekar77-sketch)
