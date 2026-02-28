import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve,
                             precision_recall_curve, classification_report)
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')

from preprocess import full_pipeline


def get_models():
    """
    Return dict of models to compare.
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost':             XGBClassifier(random_state=42, eval_metric='logloss'),
        'SVM':                 SVC(probability=True, random_state=42)
    }


def evaluate_model(model, X_test, y_test):
    """
    Return full evaluation metrics for a model.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4),
        'F1 Score':  round(f1_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
    }


def cross_validate_models(models, X_train, y_train, cv=5):
    """
    Cross-validate all models and return CV scores.
    """
    print("\n=== Cross-Validation Results ===")
    cv_results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                  cv=skf, scoring='roc_auc')
        cv_results[name] = scores
        print(f"{name:25s} | AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


def plot_roc_curves(models, X_test, y_test):
    """
    Plot ROC curves for all models.
    """
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0,1], [0,1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — Model Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('../results/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("ROC curves saved to results/roc_curves.png")


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for Random Forest.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(indices)),
            importances[indices],
            color='steelblue',
            edgecolor='white')
    plt.xticks(range(len(indices)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.title('Top 15 Feature Importances — Random Forest')
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Feature importance saved to results/feature_importance.png")


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Plot confusion matrix for best model.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('../results/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Confusion matrix saved to results/confusion_matrix.png")


def train_and_compare():
    """
    Main training pipeline — trains, evaluates, logs, and saves best model.
    """
    # Load data
    X, y, encoders, scaler = full_pipeline()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape} | Test size: {X_test.shape}")

    # Get models
    models = get_models()

    # Cross-validate
    cv_results = cross_validate_models(models, X_train, y_train)

    # Train all models
    print("\n=== Training All Models ===")
    trained_models = {}
    results = {}

    mlflow.set_experiment("ml-pipeline-comparison")

    for name, model in models.items():
        print(f"Training: {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics

        # Log to MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_params(model.get_params())
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name.replace('-', '_'), value)
            mlflow.sklearn.log_model(model, name)

        print(f"  ROC-AUC: {metrics['ROC-AUC']} | "
              f"F1: {metrics['F1 Score']} | "
              f"Precision: {metrics['Precision']} | "
              f"Recall: {metrics['Recall']}")

    # Results summary
    results_df = pd.DataFrame(results).T
    print("\n=== Final Model Comparison ===")
    print(results_df.sort_values('ROC-AUC', ascending=False))

    # Save results
    results_df.to_csv('../results/model_comparison.csv')
    print("\nResults saved to results/model_comparison.csv")

    # Plots
    plot_roc_curves(trained_models, X_test, y_test)
    plot_feature_importance(trained_models['Random Forest'],
                             list(X.columns))
    best_model_name = results_df['ROC-AUC'].idxmax()
    plot_confusion_matrix(trained_models[best_model_name],
                           X_test, y_test, best_model_name)

    # Save best model
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, '../results/best_model.pkl')
    print(f"\nBest model: {best_model_name}")
    print(f"Best ROC-AUC: {results_df['ROC-AUC'].max():.4f}")
    print("Best model saved to results/best_model.pkl")

    return trained_models, results_df


if __name__ == "__main__":
    trained_models, results_df = train_and_compare()
