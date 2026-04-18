# 🏦 Loan Approval Prediction

A machine learning project that predicts loan approval outcomes using **XGBoost**, achieving **95.5% AUC** and **95% accuracy** on 58,645 applications.

---

## 📌 Overview

Financial institutions process thousands of loan applications daily. This project builds an end-to-end ML pipeline to automate and support approval decisions — flagging high-risk applicants based on borrower profile, loan characteristics, and credit history.

---

## 📁 Project Structure

```
loan-approval-prediction/
│
├── loan_prediction.ipynb       # Main notebook (EDA + modelling)
├── train.csv                   # Training data
├── test.csv                    # Test data
├── submission.csv              # Final predictions
└── README.md
```

---

## 📊 Dataset

The dataset contains **58,645 loan applications** with the following features:

| Group | Features |
|---|---|
| **Borrower profile** | `person_age`, `person_income`, `person_emp_length`, `person_home_ownership` |
| **Loan details** | `loan_amnt`, `loan_intent`, `loan_grade`, `loan_int_rate`, `loan_percent_income` |
| **Credit history** | `cb_person_default_on_file`, `cb_person_cred_hist_length` |
| **Target** | `loan_status` (0 = not approved, 1 = approved) |

---

## 🔍 Exploratory Data Analysis

- Analysed **class distribution** to understand approval rate imbalance
- Plotted **histograms + KDE** for all numerical features to detect skew and outliers
- Used **count plots** for categorical features (loan grade, intent, home ownership)
- Identified key risk patterns: short credit history, high DTI, and low loan grades correlate strongly with rejection

---

## ⚙️ Feature Engineering

8 new financial risk indicators were created from the raw features:

| Feature | Description |
|---|---|
| `income_loan_ratio` | Income relative to loan amount |
| `dti_ratio` | Debt-to-income percentage |
| `age_group` | Borrower age bucketed into 5 brackets |
| `long_employment` | Flag for 5+ years of employment |
| `high_income` | Flag for income above $75,000 |
| `loan_grade_score` | Ordinal encoding of loan grade (A=1 → G=7) |
| `interest_to_loan` | Interest rate normalised per dollar borrowed |
| `credit_per_age` | Credit history length relative to age |

---

## 🤖 Model

**XGBoost Classifier** with a `ColumnTransformer` preprocessing pipeline:
- `StandardScaler` for numerical features
- `OneHotEncoder` for categorical features

**Training strategy:** 5-Fold Stratified Cross-Validation with early stopping on AUC

**Key hyperparameters:**

```python
XGBClassifier(
    n_estimators=2500,
    learning_rate=0.025,
    max_depth=6,
    subsample=0.88,
    colsample_bytree=0.88,
    reg_alpha=0.8,
    reg_lambda=6,
    early_stopping_rounds=100
)
```

---

## 📈 Results

| Metric | Score |
|---|---|
| OOF AUC | **0.9554** |
| Accuracy | **95%** |
| F1-Score | **0.82** |
| Best Threshold | 0.5226 |

> Threshold was tuned using the precision-recall curve on out-of-fold predictions to maximise F1-score.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-blue)

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook loan_prediction.ipynb
```

---

## 👤 Author

**[Your Name]** — Associate Data Scientist  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/your-username)
