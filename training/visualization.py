# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_model_comparison(results_dict):
    """
    Plots ROC and Precision-Recall Curves for multiple models.
    results_dict structure: {'ModelName': {'y_true': [...], 'y_probs': [...]}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. ROC Curve
    for name, data in results_dict.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_probs'])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.set_title('ROC Curve Comparison')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    for name, data in results_dict.items():
        precision, recall, _ = precision_recall_curve(data['y_true'], data['y_probs'])
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})')

    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve Comparison')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison_curves.png', dpi=300)
    print("Saved model_comparison_curves.png")


def plot_hyperparameter_heatmap(tuning_history, struct_param_name='bins'):
    """
    Generates a heatmap of Validation AUC-PR scores.
    X-axis: Structural Parameter (e.g., bins or k)
    Y-axis: GCN Hidden Channels
    """
    df = pd.DataFrame(tuning_history)

    # Pivot to matrix format for heatmap
    # We aggregate by taking the mean if multiple dropouts exist for the same hidden/struct combo
    pivot_table = df.pivot_table(values='score',
                                 index='hidden_channels',
                                 columns=struct_param_name,
                                 aggfunc='mean')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f', cbar_kws={'label': 'Mean Val AUC-PR'})

    plt.title(f'Hyperparameter Heatmap\n(GCN Hidden Channels vs {struct_param_name})')
    plt.ylabel('GCN Hidden Channels')
    plt.xlabel(f'Structural Parameter ({struct_param_name})')

    plt.savefig('hyperparameter_heatmap.png', dpi=300)
    print("Saved hyperparameter_heatmap.png")
