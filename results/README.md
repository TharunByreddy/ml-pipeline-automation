# Results Directory

## Output Files
This folder contains all generated outputs after running the pipeline:

| File | Description |
|------|-------------|
| `roc_curves.png` | ROC curves for all models |
| `feature_importance.png` | Random Forest feature importance |
| `confusion_matrix.png` | Best model confusion matrix |
| `shap_summary.png` | SHAP feature importance plot |
| `model_comparison.csv` | All model metrics comparison |
| `best_model.pkl` | Saved best performing model |
| `eda_distributions.png` | EDA distribution plots |

## How to generate
```bash
python src/train.py
```
