import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
import joblib
import warnings
warnings.filterwarnings('ignore')


def full_evaluation_report(model, X_test, y_test, model_name="Model"):
    """
    Print a complete evaluation report for a model.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*55)
    print(f"   EVALUATION REPORT — {model_name}")
    print("="*55)
    print(f"ROC-AUC Score      : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1 Score           : {f1_score(y_test, y_pred):.4f}")
    print(f"Precision          : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall             : {recall_score(y_test, y_pred):.4f}")
    print(f"Avg Precision      : {average_precision_score(y_test, y_prob):.4f}")
    print(f"Brier Score        : {brier_score_loss(y_test, y_prob):.4f}")
    print("="*55)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=['No Default', 'Default']))
    return y_pred, y_prob


def plot_roc_curve(model, X_test, y_test, model_name="Model"):
    """
    Plot ROC curve for a single model.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='steelblue',
             linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    plt.plot([0,1], [0,1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../results/roc_curve_single.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ROC-AUC: {auc:.4f}")


def plot_precision_recall(model, X_test, y_test, model_name="Model"):
    """
    Plot Precision-Recall curve.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color='coral',
             linewidth=2, label=f'AP = {avg_precision:.4f}')
    plt.fill_between(recall, precision, alpha=0.1, color='coral')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../results/precision_recall_curve.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Average Precision: {avg_precision:.4f}")


def plot_confusion_matrix(model, X_test, y_test, model_name="Model"):
    """
    Plot confusion matrix with percentages.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    axes[0].set_title(f'Confusion Matrix — {model_name}')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    axes[1].set_title(f'Confusion Matrix % — {model_name}')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('../results/confusion_matrix_detailed.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_threshold_analysis(model, X_test, y_test):
    """
    Plot precision, recall, F1 across different thresholds.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)

    precisions, recalls, f1s = [], [], []
    for thresh in thresholds:
        y_pred_t = (y_prob >= thresh).astype(int)
        precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, precisions, label='Precision', color='blue')
    plt.plot(thresholds, recalls,    label='Recall',    color='red')
    plt.plot(thresholds, f1s,        label='F1 Score',  color='green')
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Default threshold')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Classification Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../results/threshold_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Threshold analysis saved!")


def compare_models_bar(results_df):
    """
    Bar chart comparing all models across metrics.
    """
    metrics = ['ROC-AUC', 'F1 Score', 'Precision', 'Recall']
    x = np.arange(len(results_df.index))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['steelblue', 'coral', 'green', 'purple']

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric],
               width, label=metric, color=colors[i], alpha=0.8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df.index, rotation=15)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — All Metrics')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('../results/model_comparison_bar.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Model comparison chart saved!")


def shap_analysis(model, X_test, model_name="Random Forest"):
    """
    SHAP feature importance analysis.
    """
    try:
        import shap
        print(f"\nRunning SHAP analysis for {model_name}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:100])

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1], X_test[:100],
                          plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance — {model_name}')
        plt.tight_layout()
        plt.savefig('../results/shap_summary.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("SHAP analysis saved!")
    except Exception as e:
        print(f"SHAP analysis skipped: {e}")


if __name__ == "__main__":
    from preprocess import full_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Load and prep data
    X, y, encoders, scaler = full_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train a quick RF for evaluation demo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Run all evaluations
    full_evaluation_report(model, X_test, y_test, "Random Forest")
    plot_roc_curve(model, X_test, y_test, "Random Forest")
    plot_precision_recall(model, X_test, y_test, "Random Forest")
    plot_confusion_matrix(model, X_test, y_test, "Random Forest")
    plot_threshold_analysis(model, X_test, y_test)
    shap_analysis(model, X_test)
