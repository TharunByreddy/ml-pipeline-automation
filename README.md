# ⚙️ End-to-End ML Pipeline with Experiment Tracking

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Overview
A production-style automated ML pipeline covering data generation,
preprocessing, feature engineering, model training, evaluation, and
experiment tracking using MLflow. Compares 4 models on a credit default
prediction task.

---

## 🎯 Problem Statement
Manual ML workflows are slow, hard to reproduce, and difficult to compare.
This pipeline automates every step from raw data to final model evaluation
with full MLflow logging and visualization.

---

## 📁 Project Structure
```
ml-pipeline-automation/
├── src/
│   ├── preprocess.py        # Full preprocessing pipeline
│   ├── train.py             # Model training and comparison
│   └── evaluate.py          # Evaluation metrics and plots
├── notebooks/
│   └── pipeline_walkthrough.ipynb  # Full walkthrough
├── data/
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
├── results/                 # Output plots and metrics
├── requirements.txt
└── README.md
```

---

## 🔬 Pipeline Steps
1. Synthetic dataset generation (5,000 records)
2. Data quality checks and missing value imputation
3. Feature engineering — 4 new interaction features
4. Label encoding and StandardScaler normalization
5. Model training — LR, Random Forest, XGBoost, SVM
6. Cross-validation with StratifiedKFold
7. MLflow experiment tracking
8. ROC-AUC, F1, Precision, Recall evaluation
9. SHAP feature importance analysis
10. Best model saved with joblib

---

## 📊 Results

| Model | ROC-AUC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| XGBoost | 0.89 | 0.83 | 0.85 | 0.81 |
| Random Forest | 0.87 | 0.81 | 0.83 | 0.79 |
| SVM | 0.84 | 0.78 | 0.80 | 0.76 |
| Logistic Regression | 0.81 | 0.75 | 0.78 | 0.73 |

### Key Features (by importance)
| Feature | Importance |
|---------|------------|
| credit_score | 0.24 |
| debt_to_income | 0.19 |
| missed_payments | 0.17 |
| payment_history | 0.14 |
| income | 0.11 |

---

## 🛠️ Tech Stack
| Category | Tools |
|----------|-------|
| Language | Python 3.9 |
| ML Models | Scikit-learn, XGBoost |
| Tracking | MLflow |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Data | Pandas, NumPy |

---

## 🚀 How to Run
```bash
# 1. Clone the repo
git clone https://github.com/TharunByreddy/ml-pipeline-automation.git
cd ml-pipeline-automation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python src/train.py

# 4. Run evaluation only
python src/evaluate.py

# 5. View MLflow dashboard
mlflow ui

# 6. Or run the notebook
jupyter notebook notebooks/pipeline_walkthrough.ipynb
```

---

## 📬 Author
**Tharun Kumar Reddy Byreddy**
M.S. Statistical Data Science | San Francisco State University
[LinkedIn](https://www.linkedin.com/in/tharun-kumar-reddy-byeddy-801290215/) |
[GitHub](https://github.com/TharunByreddy)
